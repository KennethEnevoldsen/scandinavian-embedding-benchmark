import cohere

co = cohere.Client()

response = co.embed(
    texts=["test\n\nhello "] * 100,
    model="embed-multilingual-v3.0",
    input_type="classification",
)
print(response)
