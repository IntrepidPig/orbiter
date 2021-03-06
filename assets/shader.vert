#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;

layout (location = 0) out gl_PerVertex {
  vec4 gl_Position;
};

layout(location = 1) out vec4 frag_color;

void main() {
    frag_color = color;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
    //gl_Position = vec4(pos, 1.0);
}