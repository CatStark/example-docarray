from docarray import Document, DocumentArray
import torchvision


def preproc(d: Document):
    return (
        d.load_uri_to_image_tensor()
            .set_image_tensor_shape((180, 160))
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
    )


def reverse_preproc(d: Document):
    return (d.set_image_tensor_channel_axis(0, -1)).set_image_tensor_inv_normalization()


data_dir = "dataset"
max_files = 100

docs = DocumentArray.from_files(
    f"{data_dir}/*.jpg", size=max_files
)

# Apply pre-processing
docs.apply(preproc)

# Convert images into embeddings
model = torchvision.models.resnet50(pretrained=True)

docs.embed(model, device="cpu")  # turn this into GPU if available

# Query image
query_doc = Document(
    uri=f"{data_dir}/1.jpg"
)
query_doc.load_uri_to_image_tensor()

query_doc = preproc(query_doc)
query_doc.embed(model, device="cpu")

matches = query_doc.match(docs, limit=4).matches

matches.apply(reverse_preproc).plot_image_sprites()
