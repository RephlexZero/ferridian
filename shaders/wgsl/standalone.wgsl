struct SceneUniform {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    light_direction: vec4<f32>,
    tint_and_time: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SceneUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.world_normal = world_normal;
    output.color = input.color;
    output.world_position = world_position.xyz;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_direction = normalize(uniforms.light_direction.xyz);
    let lambert = max(dot(input.world_normal, light_direction), 0.18);
    let pulse = 0.88 + 0.12 * sin(uniforms.tint_and_time.w * 1.6 + input.world_position.y * 2.0);
    let shaded_color = input.color * uniforms.tint_and_time.xyz * lambert * pulse;
    return vec4<f32>(shaded_color, 1.0);
}