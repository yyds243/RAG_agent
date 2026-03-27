from minio import Minio
from minio.error import S3Error



def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("192.168.10.150:9000",
                   access_key="minioadmin",
                   secret_key="minioadmin",
                   secure=False
                   )

    # The file to upload, change this path if needed
    source_file = r"C:\Users\14359\Desktop\签名.jpg"

    # The destination bucket and filename on the MinIO server
    bucket_name = "test-admin"
    destination_file = "签名照片.jpg"

    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)