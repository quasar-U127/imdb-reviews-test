import tensorflow_datasets as tfds


builder = tfds.builder("imdb_reviews")
print(builder.info)

builder.download_and_prepare()

datasets = builder.as_dataset()
print(datasets)
for x in datasets.take(1):
    print(x)
