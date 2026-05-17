from __future__ import annotations

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.auth_service import insert_or_update_demo_user


def main() -> None:
    demo_users = [
        ("admin_demo", "admin@lesothohome.ai", "Admin Demo", "admin123", "admin", "Maseru HQ"),
        ("customer_demo", "user@lesothohome.ai", "Customer Demo", "user123", "customer", "Maseru"),
    ]
    for username, email, full_name, password, role, address in demo_users:
        insert_or_update_demo_user(
            username=username,
            email=email,
            full_name=full_name,
            password=password,
            role=role,
            address=address,
        )
        print(f"Seeded {role} user: {email}")


if __name__ == "__main__":
    main()
