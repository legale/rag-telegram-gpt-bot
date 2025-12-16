from src.core.access_control import AdminAccessControl


class _AdminStore:
    def __init__(self, admin):
        self._admin = admin

    def admin_exists(self):
        return self._admin is not None

    def get_admin(self):
        return self._admin


class _Config:
    def __init__(self, password: str):
        self._password = password

    @property
    def admin_password(self) -> str:
        return self._password


def test_admin_access_control_is_admin_false_when_missing():
    ac = AdminAccessControl(_AdminStore(None), _Config("p"))
    assert ac.is_admin(1) is False


def test_admin_access_control_is_admin_true_when_matches():
    ac = AdminAccessControl(_AdminStore({"user_id": 7}), _Config("p"))
    assert ac.is_admin(7) is True
    assert ac.is_admin(8) is False


def test_admin_access_control_verify_password():
    ac = AdminAccessControl(_AdminStore({"user_id": 7}), _Config("secret"))
    assert ac.verify_password("secret") is True
    assert ac.verify_password("nope") is False


def test_admin_access_control_get_admin():
    admin = {"user_id": 7}
    ac = AdminAccessControl(_AdminStore(admin), _Config("p"))
    assert ac.get_admin() == admin

