fn main() {
    cc::Build::new()
        .file("bridge/bridge.c")
        .opt_level(2)
        .compile("bridge");
}
