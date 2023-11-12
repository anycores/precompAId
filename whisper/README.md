# Whisper C++ implementation

This is an implementation of whisper from scratch in C++.
The related binaries are available on HuggingFace.

This is a proof-of-concept. Further modifications, improvements are coming.

## Available precompiled models

### Whisper-tiny AMD64

[Whisper tiny binaries](https://huggingface.co/anycores/whisper_tiny_v1.1_intel)

Binary contains:
* exe for testing the app quickly
* header and dll for building custom solutions
* main.cpp as an example, how to use the header (the exe compiled from this)
* weights.xdf (required to load into the graph, no other input required)
* audios folder, containing examples to try the application
* convert.py for creating the right input for the application from and arbitrary audio file 

#### Quick start

Example for the usage of whisper.exe:
```
whisper.exe weights.xgdf audios\voice_example1.pb
```

Example compilation (with clang from the root):
```
clang++ main.cpp win64\whisper.lib -o whisper.exe
```

Example for converting:
```
python convert.py --ipath audios\voice_example_orig1.wav --opath voice_example.pb
```

#### Implementation info

Tested on:
* windows 10
* intel i7 11th gen
* clang 16.06 as compiler

Current properties:
* fp32

## Further Notes

Improved versions will arrive regularly.
Feedbacks are wellcomed. Especially the following:
* features to be add (input format, expected output format etc.)
* devices (plan to extend for mobiles, IPUs etc.)
* models (what other models would be great to accelerate)


