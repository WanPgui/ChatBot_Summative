### ChatBot_Summative ###


### Link to colab: https://colab.research.google.com/drive/1kVaSXcivOm0telMj-_Cbkv_UKN611faA?usp=sharing


#  Creative Story Generator using T5 and TensorFlow

This project fine-tunes a T5 Transformer model on a creative writing dataset to generate coherent short stories based on prompts. It includes training, evaluation, and an interactive Gradio interface for story generation.

---

##  Project Structure

```
├── data/
│   ├── train-00000-of-00002.parquet
│   ├── train-00001-of-00002.parquet
│   └── test-00000-of-00001.parquet
├── notebooks/
│   ├── training_notebook.ipynb
│   └── evaluation_visualization.ipynb
├── checkpoints/
│   └── epoch_{xx}.weights.h5
├── app/
│   └── gradio_interface.py
├── README.md
```

---

## Dataset Preparation

1. Data is loaded from the WritingPrompts dataset stored in `.parquet` format.
2. Each story is paired with a corresponding prompt.
3. The first paragraph of each story is used as the target for generation.
4. Prompts are normalized to follow the format: `"Write a story based on: ..."`

```python
prompt_inputs = ["Write a story based on: " + p.strip() for p in prompts]
prompt_outputs = [first_paragraph.strip() for first_paragraph in stories]
```

---

## Tokenization

The T5 tokenizer converts prompts and outputs to fixed-length tensors.

* `max_input_len = 256`
* `max_target_len = 256`
* Padding is set to `max_length`.
* Labels use `-100` to ignore padded tokens during loss calculation.

---

## Model and Training

* Model used: `t5-small` 
* Optimizer: AdamW via Hugging Face’s `create_optimizer()`
* Batch size: 16
* Learning rate: `5e-5`
* Warmup steps: 500
* Epochs: 4
* Early stopping and model checkpointing included

```python
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[early_stopping, checkpoint_callback, lr_reduce]
)
```

---

##  Evaluation Metrics

After training, the model is evaluated using the following metrics:

### Accuracy and F1 Score

```python
from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_labels))
```

### Confusion Matrix & Heatmap

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d")
```

###  BLEU Score

```python
from nltk.translate.bleu_score import corpus_bleu
corpus_bleu([[ref.split()] for ref in references], [pred.split() for pred in predictions])
```

### ROUGE Score

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
```

###  BERTScore

```python
from bert_score import score
P, R, F1 = score(predictions, references, lang="en", verbose=True)
```

---

##  Real vs Predicted Output Example

```text
Prompt: Write a story based on: A ghost tries to contact their living sibling.
Real: She felt the sudden chill as she sat alone...
Generated: The lights flickered and a cold breeze rushed past her...
```

---

## Visualization

Includes training and validation loss plots and metric comparison heatmaps.

```python
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
```

---

##  Gradio Web Interface

An interactive UI to input custom prompts and generate stories. Also, it saves the history of the generated stories.

```python
import gradio as gr

def generate_story(prompt):
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    output = model.generate(input_ids, max_length=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

gr.Interface(fn=generate_story, inputs="text", outputs="text").launch()
```

---

##  How to Run

1. Upload the dataset `.parquet` files to your environment.
2. Run the preprocessing and normalization script.
3. Launch the training notebook.
4. Evaluate results using the provided evaluation scripts.
5. Start `gradio_interface.py` to run the UI. The UI is clear and very understandable to work with.

---

## History

* Originally trained using `t5-small`.
* Upgraded to `t5-base` for better coherence.
* Iteratively tuned for lower coherence score (<3.5).
* Enhanced with BLEU, ROUGE, and BERTScore metrics.
* Added training history saving, checkpoints, and UI.

---

## Future Improvements

* Add multilingual prompt generation.
* Train with a large pre-trained model.
* Train with larger datasets.
* Integrate with Hugging Face Hub.
* Include coherence score visualization.
* The colab keeps on crashing so I think one should really be careful.
