#!/bin/bash

###############################################################################
# Setup Script for Linux AI Benchmarking System
# This script performs all necessary setup steps for the project
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

###############################################################################
# Check Prerequisites
###############################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python found: $PYTHON_VERSION"
        
        # Check if Python 3.8+
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
            print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 not found. Please install pip"
        exit 1
    fi
    
    # Check CUDA (optional but recommended)
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while IFS=, read -r name memory; do
            print_info "  GPU: $name ($memory)"
        done
    else
        print_warning "NVIDIA GPU not detected (CUDA optional but recommended)"
    fi
    
    # Check git
    if command -v git &> /dev/null; then
        print_success "git found"
    else
        print_warning "git not found (optional)"
    fi
}

###############################################################################
# Create Virtual Environment
###############################################################################

setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Recreate virtual environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv venv
    
    print_success "Virtual environment created"
    print_info "To activate: source venv/bin/activate"
}

###############################################################################
# Install Dependencies
###############################################################################

install_dependencies() {
    print_header "Installing Dependencies"
    
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Run setup_venv first."
        exit 1
    fi
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    print_info "Installing dependencies from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install additional optional dependencies
    print_info "Installing optional dependencies..."
    pip install black 2>/dev/null || print_warning "black not installed (optional)"
    
    print_success "All dependencies installed"
}

###############################################################################
# Create Configuration File
###############################################################################

create_config() {
    print_header "Setting Up Configuration"
    
    CONFIG_DIR="config"
    CONFIG_FILE="$CONFIG_DIR/config.yaml"
    
    # Create config directory if it doesn't exist
    mkdir -p "$CONFIG_DIR"
    
    if [ -f "$CONFIG_FILE" ]; then
        print_warning "Configuration file already exists: $CONFIG_FILE"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing configuration"
            return
        fi
    fi
    
    print_info "Creating default configuration file..."
    cat > "$CONFIG_FILE" << 'EOF'
model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
load_in_4bit: true
device: cuda
use_data_parallelism: true  # If true, load model on each GPU separately (faster). If false, use model sharding.
num_agents: 5          # Start with 1 agent for testing cross-evaluation
max_new_tokens: 512       # Increased from 64 for complete solutions (HumanEval needs 200-500 tokens)
batch_size: 8             # GPU batching: process this many tasks together (default: 8, set to 1 to disable)
temperature: 0.25         # Lower temperature for more focused, accurate generations
top_p: 0.9                # Standard nucleus sampling
top_k: 40                 # Enable top_k filtering for better quality
repetition_penalty: 1.1   # Slight penalty to reduce repetition
do_sample: true           # Enable sampling
num_beams: 1              # Greedy decoding (no beam search)
early_stopping: true      # Stop early when EOS token is found
max_retries: 2            # Retry failed generations
num_samples: 1            # Generate single sample per task (like baseline)
enable_reflection: false  # Enable reflection/retry with corrective prompt (set to true for improved accuracy)
gpu_monitor_interval: 1
benchmark: humaneval
humaneval_limit: 5
      # use 20 for a quick smoke test
output_dir: logs
enable_tracing: true
eval_workers: 4              # Number of parallel workers for test execution
eval_timeout: 10.0           # Timeout per test in seconds
use_black_formatting: false  # Format code with black before evaluation (requires: pip install black)
verbose: false               # Set to true for detailed logging (default: false for cleaner output)
EOF
    
    print_success "Configuration file created: $CONFIG_FILE"
}

###############################################################################
# Create Directories
###############################################################################

create_directories() {
    print_header "Creating Directories"
    
    DIRS=("logs" "wandb")
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
}

###############################################################################
# Set Up Environment Variables
###############################################################################

setup_environment() {
    print_header "Setting Up Environment Variables"
    
    ENV_FILE=".env"
    
    if [ -f "$ENV_FILE" ]; then
        print_warning ".env file already exists"
    else
        print_info "Creating .env file..."
        cat > "$ENV_FILE" << 'EOF'
# Python optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Tokenizers
export TOKENIZERS_PARALLELISM=false

# W&B (optional - set if you want to use W&B)
# export WANDB_MODE=online
# export WANDB_API_KEY=your_api_key_here

# Weave (optional - set to disable if causing issues)
# export DISABLE_WEAVE=false
EOF
        print_success "Created .env file"
        print_info "To use: source .env"
    fi
}

###############################################################################
# Set Up Weights & Biases (Optional)
###############################################################################

setup_wandb() {
    print_header "Setting Up Weights & Biases (Optional)"
    
    read -p "Set up Weights & Biases? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping W&B setup"
        return
    fi
    
    source venv/bin/activate
    
    if command -v wandb &> /dev/null; then
        print_info "W&B is installed"
        print_info "To login, run: wandb login"
        print_info "Or set WANDB_API_KEY in .env file"
    else
        print_warning "W&B not installed. Install with: pip install wandb"
    fi
}

###############################################################################
# Verify Installation
###############################################################################

verify_installation() {
    print_header "Verifying Installation"
    
    source venv/bin/activate
    
    print_info "Checking Python packages..."
    
    PACKAGES=("torch" "transformers" "accelerate" "bitsandbytes" "psutil" "yaml" "wandb")
    
    for package in "${PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            VERSION=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "installed")
            print_success "$package: $VERSION"
        else
            print_error "$package: NOT INSTALLED"
        fi
    done
    
    # Check CUDA availability
    print_info "Checking CUDA availability..."
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            print_success "CUDA is available"
            GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            print_info "  GPUs detected: $GPU_COUNT"
        else
            print_warning "CUDA not available (CPU mode only)"
        fi
    fi
}

###############################################################################
# System Optimizations (Optional)
###############################################################################

setup_optimizations() {
    print_header "System Optimizations (Optional)"
    
    read -p "Apply system optimizations? (requires sudo) (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping system optimizations"
        return
    fi
    
    # Check if running as root or has sudo
    if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
        print_info "Setting CPU governor to performance mode..."
        if command -v cpupower &> /dev/null; then
            sudo cpupower frequency-set -g performance 2>/dev/null && \
                print_success "CPU governor set to performance" || \
                print_warning "Could not set CPU governor (may require root)"
        else
            print_warning "cpupower not available"
        fi
        
        print_info "Enabling GPU persistence mode..."
        if command -v nvidia-smi &> /dev/null; then
            sudo nvidia-smi -pm 1 2>/dev/null && \
                print_success "GPU persistence enabled" || \
                print_warning "Could not enable GPU persistence"
        fi
    else
        print_warning "Sudo access required for system optimizations"
        print_info "You can manually run:"
        print_info "  sudo cpupower frequency-set -g performance"
        print_info "  sudo nvidia-smi -pm 1"
    fi
}

###############################################################################
# Main Setup Function
###############################################################################

main() {
    print_header "Linux AI Benchmarking System - Setup"
    
    print_info "This script will:"
    print_info "  1. Check prerequisites"
    print_info "  2. Create virtual environment"
    print_info "  3. Install dependencies"
    print_info "  4. Create configuration file"
    print_info "  5. Create necessary directories"
    print_info "  6. Set up environment variables"
    print_info "  7. Optionally set up W&B"
    print_info "  8. Verify installation"
    print_info "  9. Optionally apply system optimizations"
    echo ""
    
    read -p "Continue with setup? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi
    
    # Run setup steps
    check_prerequisites
    setup_venv
    install_dependencies
    create_config
    create_directories
    setup_environment
    setup_wandb
    verify_installation
    setup_optimizations
    
    print_header "Setup Complete!"
    
    print_success "Setup completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Activate virtual environment:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. Load environment variables (optional):"
    echo "     source .env"
    echo ""
    echo "  3. Configure W&B (if needed):"
    echo "     wandb login"
    echo ""
    echo "  4. Run the benchmark:"
    echo "     python main.py --config config/config.yaml --num_tasks 5"
    echo ""
    echo "  5. Monitor with dashboard:"
    echo "     python metrics_dashboard.py"
    echo ""
}

# Run main function
main







