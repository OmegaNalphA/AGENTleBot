# AGENTleBot

## Introduction

AGENTleBot, a Gentle (Agent) Bot, is designed to autonomously execute tasks with integrated memory and functional capabilities. This project draws inspiration from advanced AI Agent frameworks like BabyAGI and AutoGPT, aiming to provide a solution for automated task handling.

This project is primarily an exercise in exploring autonomous AI agents.

### Features

- **Integrated Memory:** The bot retains information and learns from its interactions, enhancing its decision-making over time using a Pinecone Vector Index.
- **Multi Chain of Thought:** This bot has both internal monologues and external dialogues, allowing it to think and speak simultaneously to better accomplish tasks.
- **Customizable Objectives:** Users can set specific goals for the bot, directing its focus and actions accordingly.

## How to Use AGENTleBot

### Prerequisites

- Python 3.x
- Poetry for dependency management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AGENTleBot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd AGENTleBot
   ```
3. Install dependencies using Poetry:
   ```bash
    poetry install
   ```
4. Set up your .envrc-personal file with the necessary API keys and configurations.

### Running the Bot

- Activate the virtual environment:
  ```bash
  poetry shell
  ```
- Run the bot:
  ```bash
  python main.py "<objective>" <log_setting>
  ```
  - `<objective>`: The objective for the bot to complete.
  - `<log_setting>`: Set to true for detailed logging, false for minimal logging.

## Discussion

### Design Philosophy

AGENTleBot is built with the idea of creating a flexible, intelligent agent capable of adapting to various tasks. The framework is designed to be scalable, allowing for the addition of new features and capabilities as AI technology evolves.

### Contributions and Feedback

We welcome contributions and suggestions to improve AGENTleBot. Please feel free to raise issues or submit pull requests on our GitHub repository.

### Future Enhancements

- Integration with additional APIs for broader functionality.
- Support for OpenAI function calling.
- Enhanced learning algorithms for improved decision-making.
