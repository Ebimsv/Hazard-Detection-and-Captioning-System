from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import GenerationConfig
from PIL import Image


class BLIPCaptioner:
    def __init__(
        self, model_name="models/blip-image-captioning-base", max_new_tokens=50
    ):
        """
        Initialize BLIPCaptioner with a specified model and maximum token length.
        """
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.max_new_tokens = max_new_tokens
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP model or processor: {e}")

    def get_caption(self, image):
        """
        Generate a caption for the given image.

        Parameters:
        - image: PIL Image object.

        Returns:
        - caption: Generated caption as a string.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            generation_config = GenerationConfig(max_new_tokens=self.max_new_tokens)
            out = self.model.generate(**inputs, generation_config=generation_config)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            raise RuntimeError(f"Failed to generate caption: {e}")


class InstructBLIPCaptioner:
    def __init__(
        self, model_name="microsoft/blip2-opt-2.7b-coco-instruct", max_new_tokens=50
    ):
        """
        Initialize InstructBLIPCaptioner with a specified model and maximum token length.
        """
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.max_new_tokens = max_new_tokens
        except Exception as e:
            raise RuntimeError(f"Failed to load InstructBLIP model or processor: {e}")

    def get_caption(self, image):
        """
        Generate a caption for the given image.

        Parameters:
        - image: PIL Image object.

        Returns:
        - caption: Generated caption as a string.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            generation_config = GenerationConfig(max_new_tokens=self.max_new_tokens)
            out = self.model.generate(**inputs, generation_config=generation_config)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            raise RuntimeError(f"Failed to generate caption: {e}")


class VITGCaptioner:
    def __init__(self, model_name="facebook/vit-g", max_new_tokens=35):
        """
        Initialize VITGCaptioner with a specified model and maximum token length.
        """
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.max_new_tokens = max_new_tokens
        except Exception as e:
            raise RuntimeError(f"Failed to load VITG model or processor: {e}")

    def get_caption(self, image):
        """
        Generate a caption for the given image.

        Parameters:
        - image: PIL Image object.

        Returns:
        - caption: Generated caption as a string.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            generation_config = GenerationConfig(max_new_tokens=self.max_new_tokens)
            out = self.model.generate(**inputs, generation_config=generation_config)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            raise RuntimeError(f"Failed to generate caption: {e}")


def get_captioner(model_name, max_new_tokens=50):
    """
    Factory function to create a captioner based on the model name.

    Parameters:
    - model_name: Name of the caption model (e.g., "blip_base", "instruct_blip", "vit_g").
    - max_new_tokens: Maximum number of tokens for generated captions.

    Returns:
    - An instance of the appropriate captioner class.
    """
    try:
        if model_name == "blip_base":
            return BLIPCaptioner(max_new_tokens=max_new_tokens)
        elif model_name == "instruct_blip":
            return InstructBLIPCaptioner(max_new_tokens=max_new_tokens)
        elif model_name == "vit_g":
            return VITGCaptioner(max_new_tokens=max_new_tokens)
        else:
            raise ValueError(f"Unknown caption model: {model_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to create captioner: {e}")


if __name__ == "__main__":
    from PIL import Image

    test_image_path = "pics/dog.png"
    model_name = "blip_base"

    try:
        captioner = get_captioner(model_name, max_new_tokens=50)
        try:
            test_image = Image.open(test_image_path)
        except FileNotFoundError:
            print(f"Error: The file '{test_image_path}' was not found.")
            exit()

        # Generate caption
        caption = captioner.get_caption(test_image)
        print(f"Generated Caption: {caption}")

    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
