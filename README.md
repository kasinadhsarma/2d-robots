# 2D Bird Robots Project

## Introduction
This project aims to develop a 2D simulation of bird robots with reinforcement learning capabilities. The simulation includes various functionalities such as movement, control, sensor integration, and a digital room environment for practice. The architecture is designed to be modular and extensible, allowing for easy integration of new features and functionalities.

## Project Structure
- `environment.py`: Defines the reinforcement learning environment for the bird robots.
- `train_agent.py`: Script for training the reinforcement learning agent.
- `test_agent.py`: Script for testing the trained reinforcement learning agent.
- `config.py`: Defines configuration parameters for the project.
- `requirements.txt`: Lists the project dependencies.
- `ARCHITECTURE.md`: Provides an overview of the project architecture.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAirobotics/2D-birds.git
   cd 2D-birds
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the reinforcement learning agent:
   ```bash
   python3 train_agent.py
   ```

5. Test the trained reinforcement learning agent:
   ```bash
   python3 test_agent.py
   ```

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the Apache License, Version 2.0. See the LICENSE file for more details.
