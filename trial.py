import os

import streamlit as st


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

main_path = "."

Labels_path = os.path.join(main_path,'coco.names')
LABELS = open(Labels_path).read().strip().split("\n")
st.write(" The following items can be classified by the ml model :",LABELS)
st.write("===========================")

main_path = "C:\\Users\Jithil\Desktop\projects\yolo\yolo-st"
output_path = "C:\\Users\Jithil\Desktop\projects\yolo\yolo-st\output_path"