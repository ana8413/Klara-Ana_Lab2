from dotenv import load_dotenv
import os

load_dotenv()   # loads .env from the project root

wandb_key = os.getenv("WANDB_API_KEY")
hf_key = os.getenv("HF_TOKEN")