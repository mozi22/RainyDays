
import network
import input_pipeline

dataset = input_pipeline.parse()
iterator = dataset.make_one_shot_iterator()

input_image, resulting_image = iterator.get_next()

network.create_encoder(input_image)