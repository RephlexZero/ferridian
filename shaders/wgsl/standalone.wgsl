struct TriangleUniform {
    time_seconds: f32,
    aspect_ratio: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: TriangleUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let wave = 0.08 * sin(uniforms.time_seconds + input.position.x * 3.0);
    let rotated_x = input.position.x * cos(uniforms.time_seconds * 0.55)
        - input.position.y * sin(uniforms.time_seconds * 0.55);
    let rotated_y = input.position.x * sin(uniforms.time_seconds * 0.55)
        + input.position.y * cos(uniforms.time_seconds * 0.55);

    var output: VertexOutput;
    output.clip_position = vec4<f32>(rotated_x / uniforms.aspect_ratio, rotated_y + wave, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pulse = 0.82 + 0.18 * sin(uniforms.time_seconds * 1.7);
    return vec4<f32>(input.color * pulse, 1.0);
}