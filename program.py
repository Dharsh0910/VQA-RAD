from google.colab import drive
drive.mount('/content/drive', force_remount=True)



!pip uninstall -y numpy scipy gensim nltk -q
!rm -rf /usr/local/lib/python3.*/dist-packages/numpy* /usr/share/nltk_data /root/nltk_data /usr/local/share/nltk_data
!pip install numpy scipy gensim nltk -q

import os
import json
import random
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate,Dropout, Bidirectional
from tensorflow.keras.models import load_model
import gensim
from gensim.models import KeyedVectors


from google.colab import files
files.upload()


!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


!kaggle datasets download -d leadbest/googlenewsvectorsnegative300
!unzip googlenewsvectorsnegative300.zip

model_w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)


import os
import json
import random
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate


def load_and_split_data(json_path, train_ratio=0.7, val_ratio=0.15):
    with open(json_path, 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    N = len(data)
    return data[:int(N * train_ratio)], data[int(N * train_ratio):int(N * (train_ratio + val_ratio))], data[int(N * (train_ratio + val_ratio)):]


def tokenize_question(text):
    return word_tokenize(text.lower())

def extract_feat(tokens):
    return np.array([model_w2v[word] for word in tokens if word in model_w2v])

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

avg_pool = GlobalAveragePooling2D()(vgg_base.output)
vgg_model = Model(inputs=vgg_base.input, outputs=avg_pool)

def extract_image_feat(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    return vgg_model.predict(img_array).flatten()


def get_top_answers(data, num_ans=1000):
    counts = {}
    for item in data:
        ans = item['answer']
        if isinstance(ans, list): ans = ans[0]
        ans = str(ans).lower().strip()
        counts[ans] = counts.get(ans, 0) + 1
    return [a[1] for a in sorted([(v, k) for k, v in counts.items()], reverse=True)[:num_ans]]


from tensorflow.keras import layers

def build_vqa_model(image_feat_dim, ques_len, wordvec_dim, num_answers):
    image_input = Input(shape=(image_feat_dim,))
    ques_input = Input(shape=(ques_len, wordvec_dim))
    lstm_out = Bidirectional(LSTM(256))(ques_input)
    merged = Concatenate()([image_input, lstm_out])
    merged = Dropout(0.5)(merged)
    output = Dense(num_answers, activation='softmax')(merged)
    model = Model(inputs=[image_input, ques_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_data(data, wordvec_dim=300, max_len=21, image_dir="/content/drive/MyDrive/VQA_RAD Image Folder", atoi={}):
    X_img, X_ques, Y = [], [], []
    for item in data:
        try:
            img_path = os.path.join(image_dir, item['image_name'])
            q_feat = extract_feat(tokenize_question(item['question']))
            if len(q_feat) == 0: continue
            padded = np.zeros((max_len, wordvec_dim))
            padded[:min(max_len, len(q_feat))] = q_feat[:max_len]
            ans = item['answer']
            if isinstance(ans, list): ans = ans[0]
            y = atoi.get(str(ans).lower().strip())
            if y is None: continue
            img_feat = extract_image_feat(img_path)
            X_img.append(img_feat)
            X_ques.append(padded)
            Y.append(y)
        except: continue
    print(f"Processed: {len(X_img)}/{len(data)} ({(len(X_img)/len(data))*100:.2f}%)")
    return np.array(X_img), np.array(X_ques), np.array(Y)



!pip uninstall -y nltk
!rm -rf /usr/local/share/nltk_data /usr/share/nltk_data /usr/lib/nltk_data /root/nltk_data /usr/local/lib/python*/dist-packages/nltk*
!pip install nltk -q
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


json_path = "/content/drive/MyDrive/trainset.json"
train_data, val_data, test_data = load_and_split_data(json_path)

top_answers = get_top_answers(train_data)
atoi = {a: i for i, a in enumerate(top_answers)}
itoa = {i: a for a, i in atoi.items()}
X_img_train, X_ques_train, Y_train = prepare_data(train_data, atoi=atoi)
X_img_val, X_ques_val, Y_val = prepare_data(val_data, atoi=atoi)
X_img_test, X_ques_test, Y_test = prepare_data(test_data, atoi=atoi)


model = build_vqa_model(image_feat_dim=512, ques_len=21, wordvec_dim=300, num_answers=len(atoi))
model.fit([X_img_train, X_ques_train], Y_train, validation_data=([X_img_val, X_ques_val], Y_val), batch_size=32, epochs=100)
model.save("/content/drive/MyDrive/vqa_rad_model.h5")

# Do a quick 1-epoch dry-run just to get the 'history' structure
history = model.fit([X_img_train[:32], X_ques_train[:32]], Y_train[:32],
                    validation_data=([X_img_val[:32], X_ques_val[:32]], Y_val[:32]),
                    epochs=10, batch_size=32)


history_dict = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

epochs = 10 # Define the number of epochs here

for epoch in range(epochs):  # e.g., 50
    print(f"Epoch {epoch+1}/{epochs}")
    hist = model.fit([X_img_train, X_ques_train], Y_train,
                     validation_data=([X_img_val, X_ques_val], Y_val),
                     batch_size=32, epochs=1, verbose=1)

    # Append metrics manually
    history_dict['accuracy'].append(hist.history['accuracy'][0])
    history_dict['val_accuracy'].append(hist.history['val_accuracy'][0])
    history_dict['loss'].append(hist.history['loss'][0])
    history_dict['val_loss'].append(hist.history['val_loss'][0])

import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history_dict['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss plot
plt.plot(history_dict['loss'], label='Train Loss', marker='o')
plt.plot(history_dict['val_loss'], label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



X_img_test, X_ques_test, Y_test = prepare_data(test_data, atoi=atoi)

test_loss, test_accuracy = model.evaluate([X_img_test, X_ques_test], Y_test)
print(f"\n Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")


from IPython.display import display
from PIL import Image
import ipywidgets as widgets

def predict_answer(image_path, question):
    img_feat = extract_image_feat(image_path).reshape(1, -1)
    q_feat = extract_feat(tokenize_question(question))

    padded = np.zeros((1, 21, 300))
    padded[0, :min(21, len(q_feat))] = q_feat[:21]

    pred = model.predict([img_feat, padded])
    return itoa[np.argmax(pred)]


uploader = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False)
question_box = widgets.Textarea(
    placeholder='Enter multiple questions, one per line...',
    layout=widgets.Layout(width='100%', height='120px')
)
predict_btn = widgets.Button(description='Predict', button_style='success')
output = widgets.Output()


def on_predict_clicked(b):
    output.clear_output()
    with output:
        if not uploader.value:
            print("Please upload an image.")
            return

        file_info = next(iter(uploader.value.values()))
        file_path = '/content/uploaded_image.jpg'
        with open(file_path, 'wb') as f:
            f.write(file_info['content'])

        display(Image.open(file_path).resize((256, 256)))

        questions = [q.strip() for q in question_box.value.strip().split('\n') if q.strip()]
        if not questions:
            print("Please enter at least one question.")
            return

        print("Predicting for uploaded image:")
        for q in questions:
            ans = predict_answer(file_path, q)
            print(f"{q}\nAnswer: {ans}\n")

predict_btn.on_click(on_predict_clicked)


display(widgets.VBox([uploader, question_box, predict_btn, output]))


from IPython.display import display, clear_output
from PIL import Image
import ipywidgets as widgets
import numpy as np


def predict_answer_with_confidence(image_path, question):
    img_feat = extract_image_feat(image_path).reshape(1, -1)
    q_feat = extract_feat(tokenize_question(question))
    padded = np.zeros((1, 21, 300))
    padded[0, :min(21, len(q_feat))] = q_feat[:21]

    pred = model.predict([img_feat, padded])
    confidence = np.max(pred)
    ans = itoa[np.argmax(pred)]
    return ans, confidence

upload = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False)
img_display = widgets.Output()
chat_log = widgets.Output()
question_box = widgets.Text(placeholder="Ask a question or type 'exit'")
ask_btn = widgets.Button(description="Ask", button_style="success")


state = {'img_path': None}

def handle_upload(change):
    chat_log.clear_output()
    img_display.clear_output()
    file_info = next(iter(upload.value.values()))
    state['img_path'] = '/content/uploaded_img.jpg'
    with open(state['img_path'], 'wb') as f:
        f.write(file_info['content'])
    with img_display:
        display(Image.open(state['img_path']).resize((256, 256)))
    with chat_log:
        print("You can now start asking questions about the uploaded image.\n")

def handle_question(b):
    q = question_box.value.strip()
    if not state.get('img_path'):
        with chat_log:
            print("Please upload an image first.")
        return
    if q.lower() == 'exit':
        with chat_log:
            print(" Session ended.")
        question_box.disabled = True
        ask_btn.disabled = True
        return
    if not q:
        with chat_log:
            print("Please ask a valid question.")
        return

    answer, conf = predict_answer_with_confidence(state['img_path'], q)
    with chat_log:
        print(f"\n Question: {q}")
        print(f"Predicted Answer: {answer} (confidence: {conf:.2f})")
    question_box.value = ""

upload.observe(handle_upload, names='value')
ask_btn.on_click(handle_question)

display(widgets.VBox([
    widgets.HTML("<h3>Medical VQA Assistant</h3><b>Step 1:</b> Upload image"),
    upload,
    img_display,
    widgets.HTML("<b>Step 2:</b> Ask a question"),
    widgets.HBox([question_box, ask_btn]),
    chat_log
]))

from IPython.display import display, clear_output
from PIL import Image
import ipywidgets as widgets
import numpy as np


def predict_answer_with_confidence(image_path, question):
    img_feat = extract_image_feat(image_path).reshape(1, -1)
    q_feat = extract_feat(tokenize_question(question))
    padded = np.zeros((1, 21, 300))
    padded[0, :min(21, len(q_feat))] = q_feat[:21]

    pred = model.predict([img_feat, padded])
    confidence = np.max(pred)
    ans = itoa[np.argmax(pred)]
    return ans, confidence

upload = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False)
img_display = widgets.Output()
chat_log = widgets.Output()
question_box = widgets.Text(placeholder="Ask a question or type 'exit'")
ask_btn = widgets.Button(description="Ask", button_style="success")


state = {'img_path': None}

def handle_upload(change):
    chat_log.clear_output()
    img_display.clear_output()
    file_info = next(iter(upload.value.values()))
    state['img_path'] = '/content/uploaded_img.jpg'
    with open(state['img_path'], 'wb') as f:
        f.write(file_info['content'])
    with img_display:
        display(Image.open(state['img_path']).resize((256, 256)))
    with chat_log:
        print("You can now start asking questions about the uploaded image.\n")

def handle_question(b):
    q = question_box.value.strip()
    if not state.get('img_path'):
        with chat_log:
            print("Please upload an image first.")
        return
    if q.lower() == 'exit':
        with chat_log:
            print(" Session ended.")
        question_box.disabled = True
        ask_btn.disabled = True
        return
    if not q:
        with chat_log:
            print("Please ask a valid question.")
        return

    answer, conf = predict_answer_with_confidence(state['img_path'], q)
    with chat_log:
        print(f"\n Question: {q}")
        print(f"Predicted Answer: {answer} (confidence: {conf:.2f})")
    question_box.value = ""

upload.observe(handle_upload, names='value')
ask_btn.on_click(handle_question)

display(widgets.VBox([
    widgets.HTML("<h3>Medical VQA Assistant</h3><b>Step 1:</b> Upload image"),
    upload,
    img_display,
    widgets.HTML("<b>Step 2:</b> Ask a question"),
    widgets.HBox([question_box, ask_btn]),
    chat_log
]))




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Get predicted class labels on test set
y_pred_probs = model.predict([X_img_test, X_ques_test])
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

# Collect all answers from train_data
answer_list = [item['answer'][0] if isinstance(item['answer'], list) else item['answer'] for item in train_data]
answer_counts = Counter(answer_list)

# Create DataFrame and plot
df_counts = pd.DataFrame(answer_counts.items(), columns=['Answer', 'Count']).sort_values(by='Count', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(data=df_counts.head(20), x='Answer', y='Count', palette='crest')
plt.title('Top 20 Most Frequent Answers in Training Set')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
