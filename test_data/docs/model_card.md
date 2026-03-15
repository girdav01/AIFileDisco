# Model Card: BERT Sentiment Classifier

## Model Details
- Architecture: BERT-base-uncased
- Parameters: 110M
- Fine-tuned on: IMDB dataset

## Intended Use
Sentiment classification of English text into positive, negative, or neutral.

## Limitations
- English only
- May not generalize to domain-specific text
- Not suitable for hate speech detection

## Training Data
- IMDB movie reviews (50K samples)
- 80/10/10 train/val/test split
