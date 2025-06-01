import streamlit as st


class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def header_h4(self, text):
        self._markdown(text, "h4", " " * 6)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown(
                "\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        import re

        # Xử lý key: bỏ dấu, thay dấu cách bằng -, xử lý các ký tự đặc biệt
        key = text.lower()
        # Bỏ dấu tiếng Việt
        key = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', key)
        key = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', key)
        key = re.sub(r'[ìíịỉĩ]', 'i', key)
        key = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', key)
        key = re.sub(r'[ùúụủũưừứựửữ]', 'u', key)
        key = re.sub(r'[ỳýỵỷỹ]', 'y', key)
        key = re.sub(r'[đ]', 'd', key)

        # Thay & bằng -and-
        key = key.replace('&', '-and-')

        # Bỏ các dấu như (,) và các ký tự đặc biệt khác
        key = re.sub(r'[^\w\s-]', '', key)

        # Thay dấu cách bằng -
        key = re.sub(r'\s+', '-', key)

        # Bỏ các dấu - thừa ở đầu và cuối
        key = key.strip('-')

        # Thay nhiều dấu - liên tiếp bằng một dấu -
        key = re.sub(r'-+', '-', key)

        st.markdown(f"<{level} id='{key}'>{text}</{level}>",
                    unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")
