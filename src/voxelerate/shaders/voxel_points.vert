#version 330 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_scale;

void main() {
    vec4 world_pos = u_model * vec4(a_position, 1.0);
    vec4 view_pos = u_view * world_pos;
    gl_Position = u_projection * view_pos;
    gl_PointSize = clamp(u_point_scale / max(-view_pos.z, 0.01), 2.0, 14.0);
}
