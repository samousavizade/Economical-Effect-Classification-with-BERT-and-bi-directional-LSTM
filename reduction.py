import fasttext
from fasttext.util import reduce_model

ft = fasttext.load_model('cc.fa.300.bin', )
print("model loaded ...")

reduce_model(ft, 100)
print("embedding dimension reduced ...")

ft.save_model("cc.fa.100.bin")
print("reduced model saved ...")
