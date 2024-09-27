from dotenv import load_dotenv
import os

import cv2
import torch
import traceback
import concurrent.futures

from io import BytesIO
import numpy as np
from src.grid_recognition import detect_sudoku_grid, warp_perspective, extract_sudoku_grid_and_classify
from src.train_model import DigitClassifier
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load environment variables from .env file
load_dotenv()
# Load GPT-2 model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load Sudoku model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './models/digit_classifier.pth'

try:
    sudoku_model = DigitClassifier().to(device)
    sudoku_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Sudoku model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading Sudoku model: {e}")
    traceback.print_exc()

# Dictionary to keep conversation history for each user
user_conversations = {}

# Context for the LLM to guide its role as a Sudoku-solving bot
bot_context = """
You are a helpful and friendly chatbot designed to assist users with Sudoku puzzles.
You can solve Sudoku puzzles, explain Sudoku-related topics, and engage in general conversation.
However, always prioritize Sudoku-related responses when appropriate.
If the user sends an image of a Sudoku puzzle, you will attempt to solve it.
If the user asks unrelated questions, respond politely and guide the conversation back to Sudoku when possible.
"""


# Function to generate a response using DialoGPT with context and handling token limits
def generate_llm_response(user_id, user_message, max_new_tokens=50):
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    # Append the new message to the conversation history
    user_conversations[user_id].append(user_message)

    # Limit the conversation history to a reasonable size (for example, the last 100 tokens)
    conversation_history = " ".join(user_conversations[user_id])
    inputs = tokenizer(conversation_history, return_tensors="pt").to(device)

    # Ensure that the input is trimmed if it exceeds the model's token limit (usually 1024 for DialoGPT-medium)
    max_input_length = model.config.n_positions - max_new_tokens
    input_ids = inputs["input_ids"][:, -max_input_length:]

    # Generate response with max_new_tokens
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the generated text
    response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append the response to the conversation history
    user_conversations[user_id].append(response)

    return response


# Define function to solve Sudoku from an image in memory with a timeout
def solve_sudoku_with_timeout(image_data, model, device, timeout_duration=30):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(solve_sudoku_from_image_in_memory, image_data, model, device)
        try:
            return future.result(timeout=timeout_duration)
        except concurrent.futures.TimeoutError:
            return None, "Timeout: The Sudoku-solving process took too long. Please try again with a clearer image."


# Function to solve Sudoku from an image loaded in RAM
def solve_sudoku_from_image_in_memory(image_data, model, device):
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return None, "Error: Could not load the image. Please try again."

    try:
        grid_contour = detect_sudoku_grid(image)
        if grid_contour is not None:
            colorful_pic, bw_warped_image = warp_perspective(image, grid_contour)
            sudoku_grid, solution = extract_sudoku_grid_and_classify(colorful_pic, bw_warped_image, model, device)
            if sudoku_grid is not None:
                draw_solution_on_image(sudoku_grid, solution, colorful_pic)
                return colorful_pic, None
            else:
                return None, "Unable to solve the Sudoku puzzle. Make sure the image is clear and try again."
        else:
            return None, "Failed to detect Sudoku grid. Please ensure the image is a clear picture of a Sudoku puzzle."
    except Exception as e:
        traceback.print_exc()
        return None, f"An error occurred: {e}"


# Draw the solution on the colorful image
def draw_solution_on_image(original_grid, solved_grid, colorful_image):
    cell_height, cell_width = colorful_image.shape[:2]
    cell_height //= 9
    cell_width //= 9

    for i in range(9):
        for j in range(9):
            if original_grid[i][j] == 0 and solved_grid[i][j] != 0:
                x = j * cell_width + cell_width // 2
                y = i * cell_height + cell_height // 2
                color = (7, 7, 7)
                cv2.putText(colorful_image, str(solved_grid[i][j]), (x - 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


# Start command handler
def start(update: Update, _: CallbackContext) -> None:
    update.message.reply_text("Hello! Send me an image of a Sudoku puzzle, and I'll solve it for you!")


# Help command handler
def help_command(update: Update, _: CallbackContext) -> None:
    help_text = (
        "Here’s how you can use this bot:\n\n"
        "1. Send a clear image of a Sudoku puzzle.\n"
        "2. The bot will process the image and return the solved puzzle.\n"
        "3. You can start over at any time by sending a new image.\n\n"
        "To get started, just send an image!"
    )
    update.message.reply_text(help_text)


# About command handler
def about_command(update: Update, _: CallbackContext) -> None:
    about_text = (
        "I am a Sudoku-solving bot! You can send me a picture of a Sudoku puzzle, "
        "and I'll solve it for you. If you have any questions, just ask!"
    )
    update.message.reply_text(about_text)


# Chat handler using DialoGPT for conversations
def llm_chat_handler(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    user_id = update.message.from_user.id

    # Generate LLM response using DialoGPT
    response = generate_llm_response(user_id, user_message)

    # Send the response to the user
    update.message.reply_text(response)


# Function to handle image received from the user, processing it in memory with a timeout
def image_handler(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    update.message.reply_text(f"Thanks, {user.first_name}! I’m processing the image. This might take a few moments...")

    file = update.message.photo[-1].get_file()

    # Get the image as a byte stream (without saving to disk)
    image_data = BytesIO(file.download_as_bytearray())

    # Solve the Sudoku with a timeout
    solved_image, error = solve_sudoku_with_timeout(image_data.getvalue(), sudoku_model, device, timeout_duration=30)

    if solved_image is not None:
        # Encode the solved image to bytes
        _, buffer = cv2.imencode('.jpg', solved_image)
        solved_image_bytes = BytesIO(buffer)

        # Send the solved image back to the user
        update.message.reply_photo(photo=solved_image_bytes)
        update.message.reply_text("Sudoku solved! Here is the solution.")
    else:
        update.message.reply_text(f"Error solving Sudoku: {error}")


# Set up the bot
def main():
    TOKEN = os.getenv("TOKEN")
    updater = Updater(TOKEN)

    dispatcher = updater.dispatcher

    # Register the commands and handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("about", about_command))
    dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, llm_chat_handler))

    # Start the bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
