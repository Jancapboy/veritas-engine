#!/bin/bash
# Veritas Engine startup script

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_info "Run: cd $PROJECT_DIR && python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Check Python
if [ ! -f "$PYTHON" ]; then
    log_error "Python not found in virtual environment"
    exit 1
fi

# Export PYTHONPATH for venv packages
export PYTHONPATH="$VENV_DIR/lib/python3.13/site-packages"

case "${1:-start}" in
    start)
        log_info "Starting Veritas Engine..."
        "$PYTHON" -m veritas_engine start
        ;;
    
    stop)
        log_info "Stopping Veritas Engine..."
        pkill -f "veritas_engine" || true
        log_success "Stopped"
        ;;
    
    status)
        log_info "Checking status..."
        "$PYTHON" -m veritas_engine status
        ;;
    
    run)
        if [ -z "$2" ]; then
            log_error "Usage: $0 run <goal>"
            exit 1
        fi
        log_info "Running goal: $2"
        "$PYTHON" -m veritas_engine run "$2"
        ;;
    
    api)
        log_info "Starting API server..."
        "$PYTHON" -m veritas_engine api "${@:2}"
        ;;
    
    audit)
        log_info "Running cognitive audit..."
        "$PYTHON" -m veritas_engine audit
        ;;
    
    daemon)
        log_info "Checking daemon status..."
        "$PYTHON" -m veritas_engine daemon-status
        ;;
    
    info)
        "$PYTHON" -m veritas_engine info
        ;;
    
    milestones)
        "$PYTHON" -m veritas_engine milestones
        ;;
    
    criteria)
        "$PYTHON" -m veritas_engine criteria
        ;;
    
    install)
        log_info "Installing dependencies..."
        source "$VENV_DIR/bin/activate"
        pip install -e "$PROJECT_DIR"
        log_success "Installation complete"
        ;;
    
    test)
        log_info "Running tests..."
        source "$VENV_DIR/bin/activate"
        pytest "$PROJECT_DIR/tests" -v
        ;;
    
    help|--help|-h)
        echo "Veritas Engine - 真理引擎"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start              Start the engine"
        echo "  stop               Stop the engine"
        echo "  status             Show system status"
        echo "  run <goal>         Run a single goal inference"
        echo "  api                Start API server"
        echo "  audit              Run cognitive audit"
        echo "  daemon             Show daemon status"
        echo "  info               Show system information"
        echo "  milestones         Show development milestones"
        echo "  criteria           Show success criteria"
        echo "  install            Install dependencies"
        echo "  test               Run test suite"
        echo "  help               Show this help"
        ;;
    
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
