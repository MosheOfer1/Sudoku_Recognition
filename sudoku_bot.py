import argparse
import io
import threading
import openai
from dotenv import load_dotenv
import os
import cv2
import torch
import traceback
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

    # Limit the conversation history size
    if len(conversation_history[user_id]) > 10:  # Keep only the last 10 messages
        conversation_history[user_id] = conversation_history[user_id][-10:]


class TimeoutException(Exception):
    pass


def solve_sudoku_from_image_in_memory(image_data, model, device, stop_event):
    # Modify this function to periodically check the stop_event
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return None, "Error: Could not load the image. Please try again."
    try:
        grid_contour = detect_sudoku_grid(image)
        if grid_contour is not None:
            colorful_pic, bw_warped_image = warp_perspective(image, grid_contour)
            sudoku_grid, solution = extract_sudoku_grid_and_classify(colorful_pic, bw_warped_image, model, device, stop_event=stop_event)
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


def run_with_timeout(func, args, timeout_duration):
    result = [None]
    exception = [None]
    stop_event = threading.Event()

    def target():
        try:
            result[0] = func(*args, stop_event)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_duration)

    if thread.is_alive():
        stop_event.set()  # Signal the thread to stop
        thread.join(1)  # Wait a bit more for the thread to finish
        if thread.is_alive():
            raise TimeoutException("Function execution timed out and thread is still running")
        else:
            raise TimeoutException("Function execution timed out")

    if exception[0]:
        raise exception[0]

    return result[0]


def sudoku_handler(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user = update.message.from_user

    # Ensure the user has a conversation history
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    if not update.message.photo:
        message = "Please send a photo of a Sudoku puzzle."
        context.bot.send_message(chat_id=update.effective_chat.id, text=message)
        conversation_history[user_id].append({"role": "assistant", "content": message})
        return
    else:
        processing_message = f"Thanks, {user.first_name}! I’m processing the image. This might take a few moments..."
        context.bot.send_message(chat_id=update.effective_chat.id, text=processing_message)
        conversation_history[user_id].append({"role": "assistant", "content": processing_message})

    # Add user's action to conversation history
    conversation_history[user_id].append({"role": "user", "content": "Sent a Sudoku puzzle image"})

    file = context.bot.get_file(update.message.photo[-1].file_id)
    image_buffer = io.BytesIO()
    file.download(out=image_buffer)
    image_data = image_buffer.getvalue()

    try:
        solved_image, error_message = run_with_timeout(
            solve_sudoku_from_image_in_memory,
            (image_data, context.bot_data['model'], context.bot_data['device']),
            25  # 25 seconds timeout
        )

        if solved_image is not None:
            is_success, buffer = cv2.imencode(".jpg", solved_image)
            if is_success:
                bio = io.BytesIO(buffer)
                bio.seek(0)
                context.bot.send_photo(chat_id=update.effective_chat.id, photo=bio, caption="Here's the solved Sudoku!")
                conversation_history[user_id].append({"role": "assistant", "content": "Sent a solved Sudoku image"})
            else:
                message = "Error: Failed to encode the solved image."
                context.bot.send_message(chat_id=update.effective_chat.id, text=message)
                conversation_history[user_id].append({"role": "assistant", "content": message})
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text=error_message)
            conversation_history[user_id].append({"role": "assistant", "content": error_message})

    except TimeoutException:
        message = "Sudoku solving timed out. Please try again with a clearer image."
        context.bot.send_message(chat_id=update.effective_chat.id, text=message)
        conversation_history[user_id].append({"role": "assistant", "content": message})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        context.bot.send_message(chat_id=update.effective_chat.id, text=error_message)
        conversation_history[user_id].append({"role": "assistant", "content": error_message})
        print(f"Error in sudoku_handler: {str(e)}")
        traceback.print_exc()

    # Optionally, you can limit the conversation history size
    if len(conversation_history[user_id]) > 10:  # Keep only the last 10 messages
        conversation_history[user_id] = conversation_history[user_id][-10:]


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load Sudoku model
    device = args.device
    model_path = 'models/digit_classifier.pth'

    try:
        sudoku_model = DigitClassifier().to(device)
        sudoku_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Sudoku model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading Sudoku model: {e}")
        traceback.print_exc()
        return

    TOKEN = os.getenv("TOKEN")
    updater = Updater(TOKEN)

    dispatcher = updater.dispatcher

    # Store the model and device in bot_data for access in handlers
    dispatcher.bot_data['model'] = sudoku_model
    dispatcher.bot_data['device'] = device

    # Register the commands and handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(
        Filters.photo,
        sudoku_handler,
        run_async=True
    ))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, llm_chat_handler))

    # Start the bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku Bot')
    parser.add_argument('--device', type=str, default='cpu', help='Which device cuda/cpu')
    args = parser.parse_args()

    # Dictionary to store user conversation history
    conversation_history = {}
    openai.api_key = os.getenv("OPENAI_API_KEY")

    main()
