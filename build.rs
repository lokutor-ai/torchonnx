fn main() {
    prost_build::Config::new()
        .out_dir("src/exporter")
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .unwrap();
}
