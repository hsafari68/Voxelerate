#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;

void main() {
    vec4 color = texture(u_texture, v_uv);
    if (color.a <= 0.001) {
        discard;
    }
    frag_color = color;
}
