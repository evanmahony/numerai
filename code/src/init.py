import numerapi


def main():
    napi = numerapi.NumerAPI(verbosity="info")
    napi.download_current_dataset(dest_filename="numerai_datasets")


if __name__ == "__main__":
    main()


# evanmahony.ie
