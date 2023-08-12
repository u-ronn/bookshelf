import copy
import json
import re
import requests
from typing import List

from google.cloud import vision
from PIL import Image
from janome.tokenizer import Tokenizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud


def detect_document(image_file: bytes) -> str:
    text_ls = []
    vertices_ls = []
    credentials_dict = json.loads(st.secrets["google_credentials"])
    client = vision.ImageAnnotatorClient.from_service_account_info(info=credentials_dict)
    content = image_file.read()
    google_image = vision.Image(content=content)
    response = client.text_detection(image=google_image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                vertices = [(vertex.x, vertex.y) for vertex in paragraph.bounding_box.vertices]
                vertices_ls.append(vertices)
                word_ls = []
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    word_ls.append(word_text)
                text_ls.append(''.join(word_ls))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    google_api_df = pd.DataFrame([list(i) for i in zip(text_ls, vertices_ls)], columns=["text", "vertices"])
    google_api_df = pd.concat([google_api_df,
                               pd.DataFrame(google_api_df.vertices.tolist())],
                              axis=1)
    return google_api_df


def make_response_df(response_df: pd.DataFrame) -> pd.DataFrame:
    # 左下、右下、右上、左上
    response_df["X1"] = response_df[[0, 1, 2, 3]].apply(lambda x: (x[0][0] + x[1][0])/2, axis=1)
    response_df["Y1"] = response_df[[0, 1, 2, 3]].apply(lambda x: (x[0][1] + x[1][1])/2, axis=1)
    response_df["X2"] = response_df[[0, 1, 2, 3]].apply(lambda x: (x[2][0] + x[3][0])/2, axis=1)
    response_df["Y2"] = response_df[[0, 1, 2, 3]].apply(lambda x: (x[2][1] + x[3][1])/2, axis=1)
    response_df["a"] = (response_df["Y2"]-response_df["Y1"])/(response_df["X2"]-response_df["X1"])
    response_df["b"] = response_df["Y1"] - response_df["a"]*response_df["X1"]
    response_df["width"] = response_df[[0, 1]].apply(
        lambda x: np.sqrt(np.sum(np.square(np.array(x[0]) - np.array(x[1])))), axis=1)
    response_df["height"] = response_df[[0, 3]].apply(
        lambda x: np.sqrt(np.sum(np.square(np.array(x[0]) - np.array(x[3])))), axis=1)
    response_df["area"] = response_df["width"]*response_df["height"]
    return response_df


def response_df_check(response_df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    response_df = title_check_by_length(response_df, min_len)
    response_df = title_check_by_expr(response_df)
    response_df = title_check_by_inf(response_df)
    response_df = response_df.dropna()
    return response_df


def title_check_by_inf(response_df: pd.DataFrame) -> pd.DataFrame:
    response_df = response_df.replace([np.inf, -np.inf], np.nan)
    return response_df


def title_check_by_expr(response_df: pd.DataFrame) -> pd.DataFrame:
    response_df["title_ja"] = response_df.text.apply(lambda x: check_by_expr(x))
    return response_df


def check_by_expr(title: str) -> str:
    m = re.search(r'[ -~]*', title)
    if title == m.group():
        title = np.nan
    else:
        pass
    return title


def title_check_by_length(response_df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    response_df["title_length"] = response_df.text.apply(lambda x: x if len(x) >= min_len else np.nan)
    return response_df


def get_cluster(input_df: pd.DataFrame, n_books: int) -> pd.DataFrame:
    line_parameters = np.array([[i, j] for i, j in zip(input_df["a"], input_df["b"])])
    model = KMeans(n_clusters=n_books, random_state=0)
    model.fit(line_parameters)
    clusters = model.predict(line_parameters)
    return clusters


def make_book_df(title_list: List[str], n_books: int) -> pd.DataFrame:
    book_df_list = []
    my_bar = st.progress(0)

    for i, booktitle in enumerate(title_list):
        my_bar.progress(i / n_books)
        api_id = str(st.secrets["rakuten_api_id"])
        elements = "title,author,publisherName,salesDate,isbn,itemCaption"
        hits = 1
        url_items = f"https://app.rakuten.co.jp/services/api/BooksTotal/Search/20170404?applicationId={api_id}" \
                    f"&keyword={booktitle}&hits={hits}&elements={elements}&field=0"
        r_get = requests.get(url_items)
        items = r_get.json().get("Items")
        if items is None:
            pass
        else:
            book_search_df = pd.json_normalize(items)
            book_search_df["key"] = booktitle
            book_df_list.append(book_search_df)
    my_bar.progress(1.0)
    print(book_df_list)
    book_df = pd.concat(book_df_list)[["Item.title", "Item.author", "Item.publisherName", "Item.salesDate", "Item.isbn", "Item.itemCaption"]].rename(
        columns={
            "Item.title": "書名",
            "Item.author": "著者",
            "Item.publisherName": "出版社",
            "Item.salesDate": "発売日",
            "Item.isbn": "ISBN",
            "Item.itemCaption": "説明"}).reset_index(drop=True)
    return book_df


def create_wc_image(text_wc: str):
    t = Tokenizer()
    tokens = t.tokenize(text_wc)
    word_list = []

    for token in tokens:
        word = token.surface
        part_of_speech = token.part_of_speech.split(',')[0]
        part_of_speech_02 = token.part_of_speech.split(',')[1]

        if part_of_speech == "名詞":
            if (part_of_speech_02 != "非自立") and (part_of_speech_02 != "代名詞") and (part_of_speech_02 != "数"):
                word_list.append(word)

    words_wakati = " ".join(word_list)

    stop_words = []
    fpath = './font/ipaexg.ttf'  # 日本語フォント指定

    wordcloud = WordCloud(
        font_path=fpath,
        width=900, height=600,
        background_color="white",
        stopwords=set(stop_words),
        max_words=500,
        min_font_size=4,
        collocations=False
    ).generate(words_wakati)

    fig = plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')
    return fig


def download_csv(book_df):
    book_csv = book_df.to_csv(index=False)
    dl_click = st.download_button(label='Download file', data=book_csv, file_name="bookshelf.csv")
    return dl_click


def main():
    title = "本棚解析"
    st.set_page_config(page_title=title)
    st.title(title)
    st.caption("本棚の画像を解析します")

    if 'initial_load' not in st.session_state:
        st.session_state.initial_load = True

    if st.session_state.initial_load:
        uploaded_img = st.file_uploader("本の背表紙が写った本棚の画像をアップロードしてください")
        if uploaded_img is not None:
            st.session_state.uploaded_img_view = copy.copy(uploaded_img)
            image_view = Image.open(st.session_state.uploaded_img_view)
            st.session_state.img_array = np.array(image_view)
            st.image(st.session_state.img_array, use_column_width=True)

        st.session_state.n_books = int(st.number_input(label='写っている本の数を入力してください', min_value=0, step=1))
        st.session_state.book_df = pd.DataFrame()
        st.session_state.book_csv = st.session_state.book_df.to_csv(index=False)

        if st.session_state.n_books > 0:
            st.write('本：', st.session_state.n_books)
            response_df = detect_document(uploaded_img)
            response_df = make_response_df(response_df)
            response_df = response_df_check(response_df, min_len=1)
            response_df["cluster"] = get_cluster(response_df, st.session_state.n_books)
            book_list = response_df.groupby("cluster").aggregate({
                "area": lambda x: np.argmax(x), "text": lambda x: x.tolist()}).apply(
                lambda x: x[1][x[0]], axis=1).tolist()
            st.session_state.book_df = make_book_df(book_list, st.session_state.n_books)
            wordcloud_text = " ".join(st.session_state.book_df["説明"].tolist())
            st.dataframe(st.session_state.book_df)
            st.session_state.wordcloud_img = create_wc_image(wordcloud_text)
            st.pyplot(st.session_state.wordcloud_img)
            st.session_state.book_csv = st.session_state.book_df.to_csv(index=False)
            st.session_state.initial_load = False

    else:
        st.image(st.session_state.img_array, use_column_width=True)
        st.write('本：', st.session_state.n_books)
        st.dataframe(st.session_state.book_df)
        st.pyplot(st.session_state.wordcloud_img)

    st.download_button(label='Download file', data=st.session_state.book_csv,
                       file_name="bookshelf.csv")


if __name__ == '__main__':
    main()
