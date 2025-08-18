from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_name = "PekingU/rtdetr_r50vd"
save_path = "src/rtdetr_model"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)

processor.save_pretrained(save_path)
model.save_pretrained(save_path)
