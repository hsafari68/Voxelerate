#version 330 core

uniform vec3 u_color;
uniform float u_opacity;
out vec4 frag_color;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) {
        discard;
    }
    float alpha = smoothstep(1.0, 0.8, r2) * clamp(u_opacity, 0.0, 1.0);
    frag_color = vec4(u_color, alpha);
}
