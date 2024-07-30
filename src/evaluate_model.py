from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
import pandas as pd

def train_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    data = pd.read_csv('data/cleaned_quora_question_pairs.csv')
    inputs = tokenizer(data['question1'].tolist(), data['question2'].tolist(), return_tensors='pt', padding=True, truncation=True)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs,
        eval_dataset=inputs
    )

    trainer.train()

if __name__ == "__main__":
    train_model()
