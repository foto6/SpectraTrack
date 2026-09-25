import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "batch":
        del sys.argv[1]
        from spectratrack.batch import main as batch_main

        return batch_main()

    from spectratrack.app import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
