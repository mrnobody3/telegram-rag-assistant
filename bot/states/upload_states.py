"""FSM стани для процесу завантаження та індексації лісових документів."""

try:
    from aiogram.fsm.state import State, StatesGroup
except ImportError:

    class State:  # type: ignore[no-redef]
        """Запасний клас State для середовищ без встановленого aiogram."""

        def __set_name__(self, owner: type, name: str) -> None:
            self._state_name = f"{owner.__name__}:{name}"

        def __repr__(self) -> str:
            return getattr(self, "_state_name", "State()")

    class StatesGroup:  # type: ignore[no-redef]
        """Запасний клас StatesGroup для середовищ без встановленого aiogram."""


class UploadState(StatesGroup):
    """Група станів завантаження документа."""

    waiting_for_location = State()
