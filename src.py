import numpy as np
import pandas as pd
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Interaction content'] = df['Interaction content'].astype('U')
    df['Ticket Summary'] = df['Ticket Summary'].astype('U')
    df['y1'] = df['Type 1']
    df['y2'] = df['Type 2']
    df['y3'] = df['Type 3']
    df['y4'] = df['Type 4']
    df['x'] = df['Interaction content']
    df['y'] = df['y2']
    df = df[(df['y'] != '') & (~df['y'].isna())]
    return df

# Function to translate text to English
def trans_to_en(texts):
    t2t_m = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid", download_method=DownloadMethod.REUSE_RESOURCES)

    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l.append(text)
            continue

        doc = nlp_stanza(text)
        if doc.lang == "en":
            text_en_l.append(text)
        else:
            lang = doc.lang
            if lang == "fro": lang = "fr"
            elif lang == "la": lang = "it"
            elif lang == "nn": lang = "no"
            elif lang == "kmr": lang = "tr"

            tokenizer.src_lang = lang
            encoded_hi = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
            text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            text_en_l.append(text_en[0])

    return text_en_l

# Function to clean text data
def clean_text(df):
    noise_patterns = [
        "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    ]
    df['ts'] = df['Ticket Summary'].str.lower().replace(noise_patterns, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    
    noise_patterns_ic = [
        "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        "\d{2}(:|.)\d{2}",
        "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        "dear ((customer)|(user))",
        "dear",
        "(hello)|(hallo)|(hi )|(hi there)",
        "good morning",
        "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        "thank you for contacting us",
        "thank you for your availability",
        "thank you for providing us this information",
        "thank you for contacting",
        "thank you for reaching us (back)?",
        "thank you for patience",
        "thank you for (your)? reply",
        "thank you for (your)? response",
        "thank you for (your)? cooperation",
        "thank you for providing us with more information",
        "thank you very kindly",
        "thank you( very much)?",
        "i would like to follow up on the case you raised on the date",
        "i will do my very best to assist you",
        "in order to give you the best solution",
        "could you please clarify your request with following information:",
        "in this matter",
        "we hope you(( are)|('re)) doing ((fine)|(well))",
        "i would like to follow up on the case you raised on",
        "we apologize for the inconvenience",
        "sent from my huawei (cell )?phone",
        "original message",
        "customer support team",
        "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        "canada, australia, new zealand and other countries",
        "\d+",
        "[^0-9a-zA-Z]+",
        "(\s|^).(\s|$)"
    ]
    for noise in noise_patterns_ic:
        df['ic'] = df['Interaction content'].str.lower().replace(noise, " ", regex=True)
    df['ic'] = df['ic'].replace(r'\s+', ' ', regex=True).str.strip()
    
    return df

# Function to vectorize text data
def vectorize_text(df):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidfconverter.fit_transform(df["Interaction content"]).toarray()
    x2 = tfidfconverter.fit_transform(df["ts"]).toarray()
    X = np.concatenate((x1, x2), axis=1)
    return X

# Function to prepare training and testing data
def prepare_data(df, X):
    y = df['y'].to_numpy()
    y_series = pd.Series(y)
    good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
    y_good = y[y_series.isin(good_y_value)]
    X_good = X[y_series.isin(good_y_value)]
    y_bad = y[~y_series.isin(good_y_value)]
    X_bad = X[~y_series.isin(good_y_value)]
    
    test_size = X.shape[0] * 0.2 / X_good.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=test_size, random_state=0)
    X_train = np.concatenate((X_train, X_bad), axis=0)
    y_train = np.concatenate((y_train, y_bad), axis=0)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    p_result = pd.DataFrame(classifier.predict_proba(X_test))
    p_result.columns = classifier.classes_
    print(p_result)

# Main function to execute the workflow
def main():
    file_path = "AppGallery.csv"
    df = load_data(file_path)
    
    # Translate Ticket Summary to English
    df['ts'] = trans_to_en(df['Ticket Summary'].to_list())
    
    # Clean text data
    df = clean_text(df)
    
    # Vectorize text data
    X = vectorize_text(df)
    
    # Prepare training and testing data
    X_train, X_test, y_train, y_test = prepare_data(df, X)
    
    # Train and evaluate the model
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
