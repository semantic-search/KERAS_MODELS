from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
from db_models.models.result_model import Result
import uuid
import globals
import init
from image_recog_service import predict
import pyfiglet
import requests

global_init()


def save_to_db(db_object, result_to_save):
    print("in save")
    print(db_object)
    print(db_object.id)
    result_obj = Result()
    result_obj.results = result_to_save
    result_obj.model_name = globals.RECEIVE_TOPIC
    db_object.results.append(result_obj)
    db_object.save()

def update_state(file):
    payload = {
        'topic_name': globals.RECEIVE_TOPIC,
        'client_id': '151515',
        'value': file
    }

if __name__ == "__main__":
    print(pyfiglet.figlet_format(str(globals.RECEIVE_TOPIC)))
    print(pyfiglet.figlet_format("INDEXING CONTAINER"))
    print("Connected to Kafka at " + globals.KAFKA_HOSTNAME + ":" + globals.KAFKA_PORT)
    print("Kafka Consumer topic for this Container is " + globals.RECEIVE_TOPIC)
    for message in init.consumer_obj:
        message = message.value
        db_key = str(message)
        db_object = Cache.objects.get(pk=db_key)
        file_name = db_object.file_name
        print("#############################################")
        print("########## PROCESSING FILE " + file_name)
        print("#############################################")
        if db_object.is_doc_type:
            """document"""
            images_array = []
            for image in db_object.files:
                pdf_image = str(uuid.uuid4()) + ".jpg"
                with open(pdf_image, 'wb') as file_to_save:
                    file_to_save.write(image.file.read())
                images_array.append(pdf_image)
            to_save = list()
            for image in images_array:
                image_results = predict(image)
                to_save.append(image_results)
            save_to_db(db_object, to_save)
            print(".....................FINISHED PROCESSING FILE.....................")

        else:
            """image"""
            with open(file_name, 'wb') as file_to_save:
                file_to_save.write(db_object.file.read())
            image_results = predict(file_name)
            to_save = [image_results]
            save_to_db(db_object, to_save)
            print(".....................FINISHED PROCESSING FILE.....................")
