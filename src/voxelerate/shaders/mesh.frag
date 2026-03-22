#version 330 core

in vec3 v_world_pos;
in vec3 v_normal;

uniform vec3 u_light_pos;
uniform vec3 u_camera_pos;
uniform vec3 u_base_color;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_pos - v_world_pos);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float ambient_strength = 0.22;
    float diffuse_strength = max(dot(N, L), 0.0);
    float specular_strength = pow(max(dot(N, H), 0.0), 64.0) * 0.25;

    vec3 color = u_base_color * (ambient_strength + diffuse_strength) + vec3(specular_strength);
    frag_color = vec4(color, 1.0);
}
