from SentimentAnalyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(rnn_unit=64)

# analyzer.prepare_dataset()

# analyzer.prepare_for_training(batch_size=4)

# # analyzer.train_model(epochs=2)
# print(analyzer("hello"))
analyzer.load()
print(analyzer("ok ok"))
