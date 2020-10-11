# keras_models

## Running
- Clone this Repo
```
git clone --recurse-submodules https://github.com/semantic-search/KERAS_MODELS.git
```
- Make sure you have `.env` file with following parameters
```.env
KAFKA_HOSTNAME=
KAFKA_PORT=
MONGO_HOST=
MONGO_PORT=
MONGO_DB=
MONGO_USER=
MONGO_PASSWORD=
KAFKA_CLIENT_ID=
KAFKA_USERNAME=
KAFKA_PASSWORD=
DASHBOARD_URL=
CLIENT_ID=151515
```
- Building Dockerfile
```
docker build -t image_net_keras_models:latest
```
- Running Docker Container
> This Container requires gpu runtime in Docker
```
docker run --gpus all -it --env-file .env image_net_keras_models:latest
```
- If you want to run the prebuild dockerimage
```
docker run --gpus all -it --env-file .env ghcr.io/semantic-search/image_net_keras_models:latest
```