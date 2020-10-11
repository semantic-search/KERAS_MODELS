import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet50_decode_predictions
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.resnet import decode_predictions as resnet_decode_predictions
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess_input
from tensorflow.keras.applications.resnet_v2 import decode_predictions as resnet_v2_decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions as vgg19_decode_predictions
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.xception import decode_predictions as xception_decode_predictions
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import decode_predictions as inception_resnet_v2_decode_predictions
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions as inception_v3_decode_predictions
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from tensorflow.keras.applications.nasnet import decode_predictions as nasnet_decode_predictions
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions as mobilenet_decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as mobilenet_v2_decode_predictions


def process_image(preprocess_input, img, target_size):
    img_path = img
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def recog(decode_predictions, model, x):
    preds = model.predict(x)
    predictions = decode_predictions(preds, top=3)[0]
    labels = []
    scores = []
    for vals in predictions:
        interm = list(vals)
        interm.pop(0)
        labels.append(interm[0])
        scores.append(interm[1])
    scores = [float(np_float) for np_float in scores]
    result = {
        "labels": labels,
        "scores": scores
    }
    return result
print("*******************************************************************")
print("*******************************************************************")
print("######################### LOADING MODELS #########################")
print("*******************************************************************")
print("*******************************************************************")
resnet50_model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
print("*****************************resnet50**************************************")

resnet152_model = tf.keras.applications.ResNet152(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

print("**************************resnet152*****************************************")

resnet101_v2_model = tf.keras.applications.ResNet101V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
print("**********************resnet101_v2*********************************************")

resnet152_v2_model = tf.keras.applications.ResNet152V2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
print("**********************resnet152_v2*********************************************")
vgg16_model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
print("**********************vgg16*********************************************")
vgg19_model = tf.keras.applications.VGG19(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
print("**********************vgg19_model*********************************************")
xception_model = tf.keras.applications.Xception(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000
    )
print("**********************xception*********************************************")
inception_resnet_v2_model = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax"
    )
print("**********************inception_resnet_v2*********************************************")
inception_v3_model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
print("**********************inception_resnet_v3*********************************************")
nasnet_model = tf.keras.applications.NASNetLarge(
        input_shape=None,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )
print("**********************nasnet*********************************************")
mobilenet_model = tf.keras.applications.MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
print("**********************mobilenet*********************************************")
mobilenet_v2_model = tf.keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.4,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)
print("**********************mobilenetv2*********************************************")
print("########### FINISHED LOADING MODELS ###########")
print("*******************************************************************")
print("*******************************************************************")


def limiter(dict):
    new_labels = []
    new_scores = []
    for label, score in zip(dict["labels"], dict["scores"]):
        if score >= 0.11:
            new_labels.append(label)
            new_scores.append(score)
    final_dict = {
        "labels": new_labels,
        "scores": new_scores
    }
    return final_dict


def predict(file_name):
    x = process_image(resnet50_preprocess_input, file_name, (224, 224))
    response_dict = recog(resnet50_decode_predictions, resnet50_model, x)
    resnet_50 = limiter(response_dict)

    x = process_image(resnet_preprocess_input, file_name, (224, 224))
    response_dict = recog(resnet_decode_predictions, resnet152_model, x)
    resnet_152 = limiter(response_dict)

    x = process_image(resnet_v2_preprocess_input, file_name, (224, 224))
    response_dict = recog(resnet_v2_decode_predictions, resnet101_v2_model, x)
    resnet_101v2 = limiter(response_dict)

    response_dict = recog(resnet_v2_decode_predictions, resnet152_v2_model, x)
    resnet_152v2 = limiter(response_dict)

    x = process_image(vgg16_preprocess_input, file_name, (224, 224))
    response_dict = recog(vgg16_decode_predictions, vgg16_model, x)
    vgg16 = limiter(response_dict)

    x = process_image(vgg19_preprocess_input, file_name, (224, 224))
    response_dict = recog(vgg19_decode_predictions, vgg19_model, x)
    vgg19 = limiter(response_dict)

    x = process_image(xception_preprocess_input, file_name, (299, 299))
    response_dict = recog(xception_decode_predictions, xception_model, x)
    xception = limiter(response_dict)

    x = process_image(inception_resnet_v2_preprocess_input, file_name, (299, 299))
    response_dict = recog(inception_resnet_v2_decode_predictions, inception_resnet_v2_model, x)
    inception_resnetv2 = limiter(response_dict)

    x = process_image(inception_v3_preprocess_input, file_name, (299, 299))
    response_dict = recog(inception_v3_decode_predictions, inception_v3_model, x)
    inceptionv3 = limiter(response_dict)

    x = process_image(nasnet_preprocess_input, file_name, (331, 331))
    response_dict = recog(nasnet_decode_predictions, nasnet_model, x)
    nasnet_large = limiter(response_dict)

    x = process_image(mobilenet_preprocess_input, file_name, (224, 224))
    response_dict = recog(mobilenet_decode_predictions, mobilenet_model, x)
    mobilenet = limiter(response_dict)

    x = process_image(mobilenet_v2_preprocess_input, file_name, (224, 224))
    response_dict = recog(mobilenet_v2_decode_predictions, mobilenet_v2_model, x)
    mobilenet_large = limiter(response_dict)

    os.remove(file_name)

    full_result = {
        "results": {
            "resnet_50": resnet_50,
            "resnet_152": resnet_152,
            "resnet_101v2": resnet_101v2,
            "resnet_152v2": resnet_152v2,
            "vgg16": vgg16,
            "vgg19": vgg19,
            "xception": xception,
            "inception_resnetv2": inception_resnetv2,
            "inceptionv3": inceptionv3,
            "nasnet_large": nasnet_large,
            "mobilenet": mobilenet,
            "mobilenet_large": mobilenet_large,
        }
    }
    final_labels = list()
    final_scores = list()
    for key, value in full_result["results"].items():
        for val1, val2 in zip(value["labels"], value["scores"]):
            if val1 not in final_labels:
                final_labels.append(val1)
                final_scores.append(val2)
            else:
                x = final_labels.index(val1)
                score_to_check = final_scores[x]
                if val2 > score_to_check:
                    final_scores[x] = val2


    final_result = {
        "labels": final_labels,
        "scores": final_scores
    }
    return final_result
