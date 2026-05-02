"""State machine for OCTANE rover supervisor.

States:
    STANDBY: Initialized, no control active
    MANUAL: Remote driving via ground station
    AUTONOMOUS: RL policy navigation active
    FAULT: Emergency stop, requires reset

Transitions are triggered by mode commands and fault signals.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import rclpy.logging


class State(Enum):
    """System state machine states."""
    STANDBY = auto()
    MANUAL = auto()
    AUTONOMOUS = auto()
    FAULT = auto()


class Mode(Enum):
    """Mode commands (match ground station protocol)."""
    STANDBY = "standby"
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    FAULT_RESET = "fault_reset"


@dataclass
class StateTransition:
    """Represents a state transition event."""
    from_state: State
    to_state: State
    trigger: str
    success: bool
    reason: Optional[str] = None


class StateMachine:
    """Simple state machine for rover operation modes.

    Thread-safe state transitions with validation.
    """

    def __init__(self):
        self._state = State.STANDBY
        self._fault_active = False
        self._fault_type: Optional[str] = None
        self.logger = rclpy.logging.get_logger("state_machine")

        # Valid transitions: (from_state, to_state) -> allowed
        self._valid_transitions = {
            (State.STANDBY, State.MANUAL): True,
            (State.STANDBY, State.AUTONOMOUS): True,
            (State.STANDBY, State.FAULT): True,
            (State.MANUAL, State.STANDBY): True,
            (State.MANUAL, State.FAULT): True,
            (State.AUTONOMOUS, State.STANDBY): True,
            (State.AUTONOMOUS, State.FAULT): True,
            (State.FAULT, State.STANDBY): True,  # Only after reset + fault cleared
        }

    @property
    def state(self) -> State:
        """Current state."""
        return self._state

    @property
    def is_fault(self) -> bool:
        """True if fault state is active."""
        return self._state == State.FAULT

    @property
    def fault_type(self) -> Optional[str]:
        """Type of active fault, or None."""
        return self._fault_type

    @property
    def navigation_enabled(self) -> bool:
        """True if autonomous navigation should run."""
        return self._state == State.AUTONOMOUS

    @property
    def manual_enabled(self) -> bool:
        """True if manual control should run."""
        return self._state == State.MANUAL

    @property
    def e_suggestion(self) -> bool:
        """True if e-suggestion (software fault) is active."""
        return self._state == State.FAULT

    def transition(self, target_mode: Mode, fault_type: Optional[str] = None) -> StateTransition:
        """Attempt to transition to a new state.

        Args:
            target_mode: Desired mode
            fault_type: If provided, triggers fault transition

        Returns:
            StateTransition with success status and reason
        """
        if fault_type:
            # Fault transition from any state
            target_state = State.FAULT
            trigger = f"fault:{fault_type}"
            self._fault_active = True
            self._fault_type = fault_type
            self.logger.error(f"Fault triggered: {fault_type}")
        else:
            target_state = self._mode_to_state(target_mode)
            trigger = f"command:{target_mode.value}"

        from_state = self._state
        transition_key = (from_state, target_state)

        if transition_key not in self._valid_transitions:
            msg = f"Invalid transition: {from_state.name} -> {target_state.name}"
            self.logger.warning(msg)
            return StateTransition(
                from_state=from_state,
                to_state=target_state,
                trigger=trigger,
                success=False,
                reason=msg
            )

        # State change approved
        self._state = target_state
        if target_state == State.STANDBY:
            self._fault_active = False
            self._fault_type = None

        self.logger.info(f"State transition: {from_state.name} -> {target_state.name}")
        return StateTransition(
            from_state=from_state,
            to_state=target_state,
            trigger=trigger,
            success=True
        )

    def clear_fault(self) -> bool:
        """Clear fault state (called when faults are resolved).

        Returns:
            True if fault was cleared, False if no fault active
        """
        if self._state == State.FAULT:
            self._fault_active = False
            self._fault_type = None
            self.logger.info("Fault cleared")
            return True
        return False

    def _mode_to_state(self, mode: Mode) -> State:
        """Convert Mode enum to State."""
        if mode == Mode.STANDBY:
            return State.STANDBY
        elif mode == Mode.MANUAL:
            return State.MANUAL
        elif mode == Mode.AUTONOMOUS:
            return State.AUTONOMOUS
        elif mode == Mode.FAULT_RESET:
            # Fault reset attempts to return to standby if faults cleared
            return State.STANDBY
        return State.STANDBY
