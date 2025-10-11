# Procedural Quest Maze (Portfolio Project)

This project is a 2D maze escape game developed with Pygame, featuring NPCs powered by a Large Language Model (LLM) via the Hugging Face Transformers library. It is designed to showcase modern software architecture, including separation of concerns, event-driven programming, and dynamic content generation.

This project serves as a practical demonstration of integrating a sophisticated AI model into a real-time interactive application, with a strong focus on creating modular, scalable, and maintainable code.

## Core Features

- **Procedural Maze Generation**: A new, fully connected maze is generated at the start of each game using a randomized algorithm, ensuring unique gameplay every time.

- **LLM-Powered NPCs**: NPCs are driven by an LLM, allowing for dynamic and unscripted conversations. Their personalities and core information are defined in external Markdown files, enabling easy modification and experimentation without changing game code.

- **Dynamic Quest System**: Players must interact with NPCs to gather crucial information (e.g., the location of a treasure and its password) to complete the objective. The system tracks player knowledge to dynamically update objectives.

- **Modular & Scalable Architecture**: The codebase is intentionally structured to be modular and maintainable, demonstrating a clear **Separation of Concerns (SoC)**:
    - `controllers`: Handle all player input (`InputHandler`) and high-level interaction logic (`InteractionHandler`).
    - `renderers`: Responsible for all on-screen drawing, completely decoupled from the game logic.
    - `games`: Contains the core game state (`Game`) and state definitions (`GameState`).
    - `ui`: Manages UI overlays like menus and text boxes.
    - `configs`: Centralizes all game settings for easy tuning.
    - `clients`: Wraps external services like the LLM, providing a consistent interface to the rest of the application.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. This project uses dependencies listed in `pyproject.toml`.
    ```bash
    pip install -e .
    ```
    *Note: Heavy libraries like `torch` may require a specific version tailored to your system's hardware (e.g., CUDA support). Please install them according to your environment if needed.*

## How to Run

Execute the main script from the project root directory:

```bash
python main.py
```

## Controls

-   **Movement**: Arrow Keys (↑, ↓, ←, →)
-   **Interact**: Spacebar (when adjacent to an NPC or object)
-   **Menu Navigation**: Arrow Keys (↑, ↓) or Number Keys (1, 2, ...)
-   **Confirm**: Enter
-   **Cancel / Exit Menu**: ESC

## Reinforcement Learning Environment

This project also includes a reinforcement learning environment based on the game logic. The agent is trained to complete the full quest automatically.

### How to Run the RL Agent

1.  **Train the agent (optional):**
    If you want to train the agent from scratch, run the training script. This will take a long time.
    ```bash
    python rl/train_full_quest.py
    ```

2.  **Run the pre-trained agent:**
    To see the pre-trained PPO agent in action, run the following script:
    ```bash
    python rl/run_full_quest.py
    ```

## Project Structure

```
/
├── actors/             # Game character logic (e.g., NPC class)
│   └── prompts/        # Markdown files defining NPC personalities and knowledge
├── clients/            # Wrappers for external APIs (e.g., LLM client)
├── configs/            # Game settings (screen size, colors, fonts, etc.)
├── controllers/        # Handles player input and interaction logic
├── fonts/              # Font files used in the game
├── sprites/            # Image assets (player, walls, NPCs, etc.)
├── games/              # Core game logic and state management
├── maps/               # Maze generation logic
├── renderers/          # Screen rendering logic
├── ui/                 # UI element management (menus, text boxes)
├── main.py             # Main entry point of the application
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # This file
```