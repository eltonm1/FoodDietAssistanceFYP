from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

# The generators use the same arguments as the CLI, only as parameters
generator = GeneratorFromDict(
    count=100,
    random_skew=True,
    random_blur=True,
    background_type=0,
    )

count = 0
labels = []
for img, lbl in generator:
    # Do something with the pillow images here.
    img.save(f"dataset_trdg/{count}.png")
    count+=1
    labels.append(lbl)

with open ("dataset_trdg/label.txt", "w") as f:
    f.write("\n".join(labels))