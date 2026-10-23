import os, re, sys, itertools, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def locate_data_dir():

    cands = []
    if len(sys.argv) > 1:
        cands.append(sys.argv[1])
    cwd = os.getcwd()
    cands += [SCRIPT_DIR, cwd, os.path.join(cwd, 'ds102_kaggle'), os.path.join(cwd, 'data'),
              os.path.join(SCRIPT_DIR, 'data'), os.path.dirname(SCRIPT_DIR)]
    for d in cands:
        if os.path.exists(os.path.join(d, 'ds102_test.csv')):
            return d
    sys.exit('Khong tim thay ds102_test.csv. Hay dat model.py cung thu muc voi 3 file CSV '
             'hoac chay: python model.py <thu_muc_du_lieu>')


DATA_DIR = locate_data_dir()
print('Thu muc du lieu:', DATA_DIR)


# ----------------------------------------------------------------------------
# 1. Doc CSV an toan (xu ly loi encoding / BOM / dong hong)
# ----------------------------------------------------------------------------
def read_csv_safe(path):
    """Thu lan luot nhieu encoding, bo BOM, bo qua dong hong, sua loi font (mojibake) neu co."""
    last_err = None
    for enc in ('utf-8-sig', 'utf-8', 'utf-16', 'cp1258', 'cp1252', 'latin-1'):
        for engine in ('c', 'python'):
            try:
                df = pd.read_csv(path, encoding=enc, engine=engine,
                                 on_bad_lines='skip', encoding_errors='replace')
                if df.shape[1] >= 2:
                    df.columns = [str(c).strip().replace('﻿', '') for c in df.columns]
                    print(f'  doc {os.path.basename(path)}: encoding={enc}, engine={engine}, shape={df.shape}')
                    return fix_mojibake(df)
            except Exception as e:  # thu cach khac
                last_err = e
    raise RuntimeError(f'Khong doc duoc {path}: {last_err}')


def fix_mojibake(df):
    """Neu text tieng Viet bi loi kieu 'Ã¡', 'Ä‘' (UTF-8 bi doc nhu latin-1) thi sua lai."""
    if 'comment' not in df.columns:
        return df
    sample = ' '.join(df['comment'].astype(str).head(200))
    if re.search(r'Ã.|Ä.|áº|á»', sample):
        def fix(s):
            try:
                return s.encode('cp1252').decode('utf-8')
            except Exception:
                try:
                    return s.encode('latin-1').decode('utf-8')
                except Exception:
                    return s
        for c in ('comment', 'product_name', 'description', 'categories', 'category'):
            if c in df.columns:
                df[c] = df[c].astype(str).map(fix)
        print('  da sua loi font (mojibake)')
    return df


def find_file(name):
    for cand in (name, name.upper(), name.lower()):
        p = os.path.join(DATA_DIR, cand)
        if os.path.exists(p):
            return p
    return None


# ----------------------------------------------------------------------------
# 2. Dac trung
# ----------------------------------------------------------------------------
EMOJI_RE = re.compile('[\U0001F000-\U0001FFFF☀-➿‍]')


def clean(s):
    return re.sub(r'\s+', ' ', str(s)).strip().lower()


def meta_features(df):
    c = df['comment'].astype(str)
    f = pd.DataFrame(index=df.index)
    f['len'] = np.log1p(c.str.len())
    f['nw'] = np.log1p(c.str.split().str.len())
    f['emoji'] = c.apply(lambda x: len(EMOJI_RE.findall(x)))
    f['emoji_r'] = f['emoji'] / (c.str.len() + 1)
    f['upper_r'] = c.apply(lambda x: sum(ch.isupper() for ch in x) / (len(x) + 1))
    f['digit_r'] = c.apply(lambda x: sum(ch.isdigit() for ch in x) / (len(x) + 1))
    f['punct_r'] = c.apply(lambda x: sum((not ch.isalnum()) and (not ch.isspace()) for ch in x) / (len(x) + 1))
    f['uniq_w'] = c.apply(lambda x: len(set(x.lower().split())) / (len(x.split()) + 1))
    f['uniq_ch'] = c.apply(lambda x: len(set(x.lower())) / (len(x) + 1))
    f['nl'] = c.str.count('\n')
    f['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(5)
    f['sold'] = np.log1p(pd.to_numeric(df['num_sold'], errors='coerce').fillna(0).clip(lower=0))
    f['nrev'] = np.log1p(pd.to_numeric(df['num_review'], errors='coerce').fillna(0).clip(lower=0))

    def overlap(row):
        a = set(clean(row['comment']).split())
        b = set(clean(row['product_name']).split()) | set(clean(row['description'])[:500].split())
        return len(a & b) / (len(a) + 1)
    f['overlap'] = df.apply(overlap, axis=1)
    return f


class Featurizer:
    def fit(self, df):
        t = df['comment'].map(clean)
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000).fit(t)
        self.c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=150000).fit(t)
        self.sc = StandardScaler().fit(meta_features(df))
        return self

    def transform(self, df):
        t = df['comment'].map(clean)
        return sp.hstack([self.w.transform(t), self.c.transform(t),
                          sp.csr_matrix(self.sc.transform(meta_features(df)))]).tocsr()


def make_model():
    return LogisticRegression(C=10, class_weight='balanced', max_iter=3000)


def tune_class_weights(y, P):
    """Tim he so nhan len xac suat tung lop de toi uu F1-macro (luoi thua de tranh overfit)."""
    grid = [0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]
    best_s, best_w = f1_score(y, P.argmax(1), average='macro'), np.ones(4)
    for w1, w2, w3 in itertools.product(grid, grid, grid):
        w = np.array([1, w1, w2, w3])
        s = f1_score(y, (P * w).argmax(1), average='macro')
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


# ----------------------------------------------------------------------------
# 3. Chay
# ----------------------------------------------------------------------------
def main():
    print('Doc du lieu...')
    p_train, p_dev, p_test = find_file('ds102_train.csv'), find_file('ds102_dev.csv'), find_file('ds102_test.csv')
    if p_test is None or (p_train is None and p_dev is None):
        sys.exit(f'Thieu file trong "{DATA_DIR}" (can ds102_test.csv va it nhat ds102_train.csv hoac ds102_dev.csv)')
    train = read_csv_safe(p_train) if p_train else None
    dev = read_csv_safe(p_dev) if p_dev else None
    test = read_csv_safe(p_test)
    test = test.rename(columns={c: 'id' for c in test.columns if c.lower() == 'id'})

    for d in (train, dev):
        if d is not None:
            d['spam_label'] = pd.to_numeric(d['spam_label'], errors='coerce')
            d.dropna(subset=['spam_label'], inplace=True)
            d['spam_label'] = d['spam_label'].astype(int)

    # --- Danh gia: train -> dev (neu co ca hai), nguoc lai CV 5-fold tren du lieu co nhan
    if train is not None and dev is not None:
        ft = Featurizer().fit(train)
        m = make_model().fit(ft.transform(train), train['spam_label'])
        P = m.predict_proba(ft.transform(dev))
        y = dev['spam_label'].values
        print('\n[Dev] F1-macro (argmax): %.4f' % f1_score(y, P.argmax(1), average='macro'))
        w, s = tune_class_weights(y, P)
        print('[Dev] F1-macro sau khi chinh he so lop %s: %.4f' % (np.round(w, 2), s))
        print(classification_report(y, (P * w).argmax(1), digits=3, zero_division=0))
    else:
        lab = train if train is not None else dev
        y = lab['spam_label'].values
        P = np.zeros((len(lab), 4))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(lab, y):
            ft = Featurizer().fit(lab.iloc[tr])
            m = make_model().fit(ft.transform(lab.iloc[tr]), y[tr])
            P[va] = m.predict_proba(ft.transform(lab.iloc[va]))
        print('\n[CV 5-fold] F1-macro (argmax): %.4f' % f1_score(y, P.argmax(1), average='macro'))
        w, s = tune_class_weights(y, P)
        print('[CV 5-fold] F1-macro sau khi chinh he so lop %s: %.4f' % (np.round(w, 2), s))
        print(classification_report(y, (P * w).argmax(1), digits=3, zero_division=0))

    # --- Huan luyen lai tren toan bo du lieu co nhan (train + dev) va du doan test
    full = pd.concat([d for d in (train, dev) if d is not None], ignore_index=True)
    ft = Featurizer().fit(full)
    m = make_model().fit(ft.transform(full), full['spam_label'].values)
    pred = (m.predict_proba(ft.transform(test)) * w).argmax(1)

    sub = pd.DataFrame({'ID': test['id'], 'spam_label': pred})
    out = os.path.join(DATA_DIR, 'submission.csv')
    sub.to_csv(out, index=False)
    print(f'\nDa luu {out}  shape={sub.shape}')
    print('Ti le nhan du doan:', sub['spam_label'].value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == '__main__':
    main()