import multiprocessing
import time

import openai
from dotenv import load_dotenv
import os
import cv2
import torch
import traceback
from io import BytesIO
import numpy as np
from src.grid_recognition import detect_sudoku_grid, warp_perspective, extract_sudoku_grid_and_classify
from src.train_model import DigitClassifier
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


def generate_llm_response(user_id, user_message):
    # Initialize or update the conversation history for the user
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add the user's latest message to the conversation history
    conversation_history[user_id].append({"role": "user", "content": user_message})

    # System message to explain the bot's purpose
    system_message = {
        "role": "system",
        "content": "You are a bot that helps users solve Sudoku puzzles. "
                   "The user sends an image of a Sudoku puzzle, "
                   "and you respond with the solved puzzle image."
    }

    # Prepare the conversation history with the system message at the start
    messages = [system_message] + conversation_history[user_id]

    # Generate a response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the generated message from the response
    assistant_message = response['choices'][0]['message']['content']

    # Add the assistant's response to the conversation history
    conversation_history[user_id].append({"role": "assistant", "content": assistant_message})

    # Return the assistant's response
    return assistant_message


def solve_sudoku_with_timeout(image_data, model, device, timeout_duration):
    # Create a multiprocessing Queue to capture the result
    result_queue = multiprocessing.Queue()

    # Create a multiprocessing process for the solver
    process = multiprocessing.Process(
        target=solve_sudoku_from_image_in_memory,
        args=(image_data, model, device, result_queue)
    )
    process.start()

    # Wait for the process to complete within the timeout
    process.join(timeout_duration)

    if process.is_alive():
        print("Timeout exceeded. Terminating the process.")
        process.terminate()  # Kill the process if it's still running
        return None, "Timeout: The Sudoku-solving process took too long. Please try again with a clearer image."
    else:
        # Retrieve the result from the queue
        if not result_queue.empty():
            result, error_message = result_queue.get()
            return result, error_message
        else:
            return None, "No result returned from the process."


# Function to solve Sudoku from an image loaded in RAM
def solve_sudoku_from_image_in_memory(image_data, model, device, result_queue):
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        result_queue.put((None, "Error: Could not load the image. Please try again."))
        return

    try:
        grid_contour = detect_sudoku_grid(image)
        if grid_contour is not None:
            colorful_pic, bw_warped_image = warp_perspective(image, grid_contour)
            sudoku_grid, solution = extract_sudoku_grid_and_classify(colorful_pic, bw_warped_image, model, device)
            if sudoku_grid is not None:
                draw_solution_on_image(sudoku_grid, solution, colorful_pic)
                result_queue.put((colorful_pic, None))  # Return result through queue
            else:
                result_queue.put((None, "Unable to solve the Sudoku puzzle. Make sure the image is clear and try again."))
        else:
            result_queue.put((None, "Failed to detect Sudoku grid. Please ensure the image is a clear picture of a Sudoku puzzle."))
    except Exception as e:
        traceback.print_exc()
        result_queue.put((None, f"An error occurred: {e}"))


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
def start(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    start_message = "Hello! Send me an image of a Sudoku puzzle, and I'll solve it for you!"

    update.message.reply_text(start_message)

    # Update the conversation history
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add the bot's message to the conversation history
    conversation_history[user_id].append({"role": "assistant", "content": start_message})


# Help command handler
def help_command(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    help_text = (
        "Here’s how you can use this bot:\n\n"
        "1. Send a clear image of a Sudoku puzzle.\n"
        "2. The bot will process the image and return the solved puzzle.\n"
        "3. You can start over at any time by sending a new image.\n\n"
        "To get started, just send an image!"
    )

    update.message.reply_text(help_text)

    # Update the conversation history
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add the bot's message to the conversation history
    conversation_history[user_id].append({"role": "assistant", "content": help_text})


# Chat handler using Llama for conversations
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
    user_id = user.id
    processing_message = f"Thanks, {user.first_name}! I’m processing the image. This might take a few moments..."

    update.message.reply_text(processing_message)

    # Update the conversation history
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add the bot's message to the conversation history
    conversation_history[user_id].append({"role": "assistant", "content": processing_message})

    file = update.message.photo[-1].get_file()

    # Get the image as a byte stream (without saving to disk)
    image_data = BytesIO(file.download_as_bytearray())

    # Start timing the Sudoku solving process
    start_time = time.time()

    # Solve the Sudoku with a timeout
    solved_image, error = solve_sudoku_with_timeout(image_data.getvalue(), sudoku_model, device, timeout_duration=5)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    if solved_image is not None:
        # Encode the solved image to bytes
        _, buffer = cv2.imencode('.jpg', solved_image)
        solved_image_bytes = BytesIO(buffer)

        # Send the solved image back to the user
        update.message.reply_photo(photo=solved_image_bytes)

        # Create a message with the time it took to solve
        solved_message = f"Sudoku solved! Here is the solution.\nIt took {elapsed_time:.2f} seconds to solve."
        update.message.reply_text(solved_message)

        # Add the bot's response to the conversation history
        conversation_history[user_id].append({"role": "assistant", "content": solved_message})
    else:
        error_message = f"Error solving Sudoku: {error}"
        update.message.reply_text(error_message)

        # Add the error message to the conversation history
        conversation_history[user_id].append({"role": "assistant", "content": error_message})


# Set up the bot
def main():
    TOKEN = os.getenv("TOKEN")
    updater = Updater(TOKEN)

    dispatcher = updater.dispatcher

    # Register the commands and handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, llm_chat_handler))

    # Start the bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Load environment variables from .env file
    load_dotenv()

    # Load Sudoku model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/digit_classifier.pth'

    try:
        sudoku_model = DigitClassifier().to(device)
        sudoku_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Sudoku model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading Sudoku model: {e}")
        traceback.print_exc()

    # Dictionary to store user conversation history
    conversation_history = {}
    openai.api_key = os.getenv("OPENAI_API_KEY")

    main()
