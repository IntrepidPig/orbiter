use std::{collections::HashMap, io::Read, path::Path, fs::File};

#[allow(unused_imports)]
use hal::{
	adapter::Adapter,
	buffer,
	command::{self, ClearColor, ClearValue, CommandBuffer},
	format::{self, AsFormat, Aspects, ChannelType, Component, Format, Rgba8Srgb, Swizzle},
	image::{self, Extent, Layout, SubresourceRange, ViewKind},
	memory,
	pass::{
		self, Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass,
		SubpassDependency, SubpassDesc,
	},
	pool::{self, CommandPool, CommandPoolCreateFlags},
	pso::{
		self, BlendState, ColorBlendDesc, ColorMask, EntryPoint, GraphicsPipelineDesc,
		GraphicsShaderSet, PipelineStage, Rasterizer, Rect, ShaderStageFlags, Specialization,
		SpecializationConstant, Viewport,
	},
	queue::{self, family::QueueGroup, Submission},
	window::Extent2D,
	Backbuffer, Backend, DescriptorPool, Device, FrameSync, Graphics, Instance, PhysicalDevice,
	Primitive, Surface, Swapchain, SwapchainConfig,
};

use gfx_backend_vulkan as backend;

use log::{debug, error, info, trace, warn};
use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent, VirtualKeyCode, ElementState};

use spa::prelude::*;

pub mod td;

// TODO: figure out how vulkan-tutorial.com "swapchain images" fit into here to be able to send a uniform
// buffer to the place.
fn main() {
	unsafe {
		init();
	}
}

fn setup_logging() {
	fern::Dispatch::new()
		.format(|out, message, record| {
			out.finish(format_args!(
				"{}[{}][{}] {}",
				chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
				record.target(),
				record.level(),
				message
			))
		})
		.level(log::LevelFilter::Info)
		.chain(std::io::stdout())
		.apply()
		.unwrap();
}

pub unsafe fn init() {
	setup_logging();
	debug!("Starting");
	let mut events_loop = EventsLoop::new();
	let window = WindowBuilder::new()
		.with_dimensions(winit::dpi::LogicalSize::new(1024 as _, 768 as _))
		.build(&events_loop)
		.unwrap();
	let instance = backend::Instance::create("SpaC", 1);
	let surface = instance.create_surface(&window);
	let mut renderer = Renderer::new(instance, window, surface);
	debug!("Started, allocating memory");
	let teapot = load_teapot();
	let (vertex_buffer, vertex_memory) = renderer.create_vertex_buffer(&teapot);
	debug!("Allocated memory, starting main loop");
	
	struct Controls {
		w: bool,
		a: bool,
		s: bool,
		d: bool,
	}
	
	let mut controls = Controls {
		w: false,
		a: false,
		s: false,
		d: false
	};
	
	let mut pos = Point3::from([0.0, 0.0, -1.0]);

	let mut running = true;

	while running {
		let start_time = std::time::Instant::now();
		trace!("loop, polling events");
		events_loop.poll_events(|event| {
			trace!("event: {:?}", event);
			match event {
				Event::WindowEvent { event, .. } => match event {
					WindowEvent::Resized(new_size) => {
						let new_size = new_size.to_physical(1.0);
						renderer.recreate_swapchain = true;
						renderer.recreate_swapchain_dims =
							Some((new_size.width as _, new_size.height as _));
					}
					WindowEvent::CloseRequested => {
						running = false;
					},
					_ => {},
				},
				Event::DeviceEvent { event, .. } => match event {
					winit::DeviceEvent::Key(input) => {
						let winit::KeyboardInput { state, virtual_keycode, modifiers: _modifiers, scancode: _scancode } = input;
						match (virtual_keycode, state) {
							(Some(VirtualKeyCode::W), ElementState::Pressed) => {
								controls.w = true;
							},
							(Some(VirtualKeyCode::A), ElementState::Pressed) => {
								controls.a = true;
							},
							(Some(VirtualKeyCode::S), ElementState::Pressed) => {
								controls.s = true;
							},
							(Some(VirtualKeyCode::D), ElementState::Pressed) => {
								controls.d = true;
							},
							(Some(VirtualKeyCode::W), ElementState::Released) => {
								controls.w = false;
							},
							(Some(VirtualKeyCode::A), ElementState::Released) => {
								controls.a = false;
							},
							(Some(VirtualKeyCode::S), ElementState::Released) => {
								controls.s = false;
							},
							(Some(VirtualKeyCode::D), ElementState::Released) => {
								controls.d = false;
							},
							_ => {},
						}
					},
					_ => {}
				},
				_ => {}
			}
		});
		
		if controls.w {
			pos.z += 0.1;
		} else if controls.a {
			pos.x += 0.1;
		} else if controls.s {
			pos.z -= 0.1;
		} else if controls.d {
			pos.x -= 0.1;
		}
		
		trace!("rendering vertices");
		let aspect = renderer.get_dims().0 as f32 / renderer.get_dims().1 as f32;
		let mvp = td::ModelViewProjection {
			model: Mat4::new_rotation(Vec3::zeros()),
			view: Mat4::look_at_rh(&pos, &Point3::from([0.0, 0.0, 0.0]), &Vec3::from([0.0, 1.0, 0.0])),
			proj: Mat4::new_perspective(aspect, 90.0, 0.0, 1000.0),
		};
		renderer.render_vertices(vertex_buffer, mvp);
		std::thread::sleep(std::time::Duration::from_millis(16));
		let dur = start_time.elapsed();
	}
}

fn load_teapot() -> Vec<td::Vertex> {
	use palette::{Hsv, RgbHue, rgb::Rgb};
	
	let mut data = Vec::new();
	let mut buf = String::new();
	let mut file = File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.dat")).unwrap();
	file.read_to_string(&mut buf).unwrap();
	
	for float_str in buf.split(',') {
		let float_str = float_str.trim();
		if float_str.is_empty() {
			continue;
		}
		
		data.push(float_str.parse::<f32>().unwrap_or_else(|_| {
			error!("Couldn't parse '{}' as float", float_str);
			0.0
		}));
	}
	
	let mut verts = Vec::with_capacity(data.len() / 3);
	for i in 0..(data.len() / 3) {
		let i = i * 3;
		verts.push(td::Vertex { pos: Point3::from([data[i], data[i+1], data[i+2]]), col: Vec4::from([0.0, 0.0, 0.0, 1.0]) });
	}
	
	let dh = 360.0 / verts.len() as f32;
	let mut deg = 0.0;
	
	for vert in &mut verts {
		let rgb: Rgb = Hsv::new(RgbHue::from_degrees(deg), 1.0, 1.0).into();
		vert.col[0] = rgb.red;
		vert.col[1] = rgb.green;
		vert.col[2] = rgb.blue;
		
		deg += dh;
	}
	
	verts
}

pub struct Renderer<B, I>
where
	B: hal::Backend,
	I: hal::Instance<Backend = B>,
{
	window: Window,
	#[allow(dead_code)]
	instance: I,
	surface: B::Surface,
	adapter: Adapter<B>,
	device: B::Device,
	queue_group: QueueGroup<B, Graphics>,
	command_pool: CommandPool<B, Graphics>,
	format: Format,
	render_pass: B::RenderPass,
	pipeline: B::GraphicsPipeline,
	pipeline_layout: B::PipelineLayout,
	mvp_buffer: B::Buffer,
	mvp_memory: B::Memory,
	swapchain: B::Swapchain,
	swapchain_images: Vec<B::Image>,
	dims: (u32, u32),
	#[allow(dead_code)]
	descriptor_pool: B::DescriptorPool,
	descriptor_sets: Vec<B::DescriptorSet>,
	frame_views: Vec<B::ImageView>,
	framebuffers: Vec<B::Framebuffer>,
	frame_semaphore: B::Semaphore,
	frame_fence: B::Fence,
	present_semaphore: B::Semaphore,
	viewport: Viewport,
	recreate_swapchain: bool,
	recreate_swapchain_dims: Option<(u32, u32)>,
}

impl<B, I> Renderer<B, I>
where
	B: Backend,
	I: Instance<Backend = B>,
{
	pub fn new(instance: I, window: Window, surface: B::Surface) -> Self {
		const WIDTH: u32 = 1024;
		const HEIGHT: u32 = 768;
		let mut adapter = Self::initialize(&instance);
		let mut surface = surface;
		let (device, queue_group) = Self::get_device_and_queue_group(&adapter, &surface);
		let command_pool = Self::create_command_pool(&device, &queue_group);
		let (_caps, formats, _present_modes, _composite_alphas) =
			surface.compatibility(&mut adapter.physical_device);
		debug!("Supported formats: {:?}", formats);
		let format = formats.map_or(Format::Rgba8Srgb, |formats| {
			formats
				.iter()
				.find(|format| format.base_format().1 == ChannelType::Srgb)
				.map(|format| *format)
				.unwrap_or(formats[0])
		});
		debug!("Picked {:?}", format);
		let render_pass = Self::build_render_pass(&device, format);
		let vert_spirv = Self::compile_shader_file(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/shader.vert"), shaderc::ShaderKind::Vertex);
		let frag_spirv = Self::compile_shader_file(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/shader.frag"), shaderc::ShaderKind::Fragment);
		let shaders = Self::load_spirv_shaders(
			&device,
			vec![
				/*(String::from("vertex"), include_bytes!("../assets/vert.spv") as &[u8]),
				(String::from("fragment"), include_bytes!("../assets/frag.spv") as &[u8]),
				*/(String::from("vertex"), vert_spirv.as_binary_u8()),
				(String::from("fragment"), frag_spirv.as_binary_u8()),
			],
		);
		let (swapchain, backbuffer) =
			Self::build_swapchain(&device, &mut adapter, &mut surface, format, None, WIDTH, HEIGHT);
		let (mvp_buffer, mvp_memory) = Self::create_uniform_buffer::<td::ModelViewProjection>(&device, &adapter);
		let (swapchain_images, frame_views, framebuffers) =
			Self::get_frame_views_and_buffers(&device, &render_pass, backbuffer, WIDTH, HEIGHT);
		let mut descriptor_pool = Self::create_descriptor_pool(&device, &swapchain_images);
		let (pipeline, pipeline_layout, descriptor_sets) = Self::build_pipeline::<td::Vertex>(&device, &render_pass, shaders, &mut descriptor_pool, &mvp_buffer);
		let (frame_semaphore, present_semaphore) = Self::create_semaphore_and_fence(&device);
		let frame_fence = Self::create_fence(&device);
		let viewport = Self::create_viewport(WIDTH, HEIGHT);
		let recreate_swapchain = false;
		let recreate_swapchain_dims = None;
		let dims = window.get_inner_size().unwrap().to_physical(1.0);
		let dims = (dims.width as _, dims.height as _);
		
		Self {
			window,
			instance,
			surface,
			adapter,
			device,
			queue_group,
			command_pool,
			format,
			render_pass,
			pipeline,
			pipeline_layout,
			swapchain,
			swapchain_images,
			descriptor_sets,
			mvp_buffer,
			mvp_memory,
			descriptor_pool,
			frame_views,
			framebuffers,
			frame_semaphore,
			frame_fence,
			present_semaphore,
			viewport,
			recreate_swapchain,
			recreate_swapchain_dims,
			dims,
		}
	}

	pub fn initialize(instance: &I) -> Adapter<B> {
		let adapter = instance.enumerate_adapters().remove(0);
		adapter
	}

	pub fn get_device_and_queue_group(
		adapter: &Adapter<B>,
		surface: &dyn Surface<B>,
	) -> (B::Device, QueueGroup<B, Graphics>) {
		adapter
			.open_with::<_, Graphics>(1, |family| surface.supports_queue_family(family))
			.unwrap()
	}

	pub fn create_command_pool(
		device: &B::Device,
		queue_group: &QueueGroup<B, Graphics>,
	) -> CommandPool<B, Graphics> {
		unsafe {
			device
				.create_command_pool_typed(queue_group, CommandPoolCreateFlags::empty())
				.unwrap()
		}
	}

	pub fn build_render_pass(device: &B::Device, format: Format) -> B::RenderPass {
		let color_attachment = Attachment {
			format: Some(format),
			samples: 1,
			ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
			stencil_ops: AttachmentOps::DONT_CARE,
			layouts: Layout::Undefined..Layout::Present,
		};

		let subpass = SubpassDesc {
			colors: &[(0, Layout::ColorAttachmentOptimal)],
			depth_stencil: None,
			inputs: &[],
			resolves: &[],
			preserves: &[],
		};

		let dependency = SubpassDependency {
			passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
			stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
			accesses: image::Access::empty()
				..(image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE),
		};

		unsafe {
			device
				.create_render_pass(&[color_attachment], &[subpass], &[dependency])
				.unwrap()
		}
	}
	
	pub fn compile_shader_file<P: AsRef<Path>>(path: P, kind: shaderc::ShaderKind) -> shaderc::CompilationArtifact {
		let mut compiler = shaderc::Compiler::new().unwrap();
		let mut file = File::open(path.as_ref()).unwrap();
		let mut source_buf = String::new();
		file.read_to_string(&mut source_buf).unwrap();
		let name = path.as_ref().file_name().unwrap().to_string_lossy().to_owned();
		compiler.compile_into_spirv(&source_buf, kind, &name, "main", None).unwrap()
	}

	pub fn load_spirv_shaders<In, R>(
		device: &B::Device,
		inputs: In,
	) -> HashMap<String, B::ShaderModule>
	where
		In: IntoIterator<Item = (String, R)>,
		R: Read,
	{
		inputs
			.into_iter()
			.map(|(name, mut input)| {
				let mut buf = Vec::new();
				input.read_to_end(&mut buf).unwrap();
				(name, unsafe { device.create_shader_module(&buf).unwrap() })
			})
			.collect()
	}
	
	pub fn create_descriptor_pool(device: &B::Device, swapchain_images: &[B::Image]) -> B::DescriptorPool {
		let descriptor_pool = unsafe { device.create_descriptor_pool(1, &[pso::DescriptorRangeDesc {
			ty: pso::DescriptorType::UniformBuffer,
			count: swapchain_images.len(),
		}]).unwrap() };
		descriptor_pool
	}
	
	pub fn create_descriptor_sets(device: &B::Device) {
	
	}

	pub fn build_pipeline<V>(
		device: &B::Device,
		render_pass: &B::RenderPass,
		shader_modules: HashMap<String, B::ShaderModule>,
		descriptor_pool: &mut B::DescriptorPool,
		mvp_buffer: &B::Buffer,
	) -> (B::GraphicsPipeline, B::PipelineLayout, Vec<B::DescriptorSet>) {
		let desc_set_layout_binding = pso::DescriptorSetLayoutBinding {
			binding: 0,
			ty: pso::DescriptorType::UniformBuffer,
			count: 1,
			stage_flags: pso::ShaderStageFlags::VERTEX,
			immutable_samplers: false
		};
		
		let desc_set_layout = unsafe { device.create_descriptor_set_layout(&[desc_set_layout_binding], &[]).unwrap() };
		
		let mut desc_sets = Vec::new();
		unsafe { descriptor_pool.allocate_sets(&[desc_set_layout], &mut desc_sets).unwrap() };
		
		unsafe {
			device.write_descriptor_sets(vec![pso::DescriptorSetWrite {
				set: &desc_sets[0],
				binding: 0,
				array_offset: 0,
				descriptors: &[pso::Descriptor::Buffer(mvp_buffer, None..None)]
			}]);
		}
		
		let desc_set_layout_binding = pso::DescriptorSetLayoutBinding {
			binding: 0,
			ty: pso::DescriptorType::UniformBuffer,
			count: 1,
			stage_flags: pso::ShaderStageFlags::VERTEX,
			immutable_samplers: false
		};
		
		let desc_set_layout = unsafe { device.create_descriptor_set_layout(&[desc_set_layout_binding], &[]).unwrap() };
		
		let pipeline_layout = unsafe { device.create_pipeline_layout(&[desc_set_layout], &[]).unwrap() };

		let vs_entry = EntryPoint::<B> {
			entry: "main",
			module: shader_modules.get("vertex").unwrap(),
			specialization: Specialization::default(),
		};

		let fs_entry = EntryPoint::<B> {
			entry: "main",
			module: shader_modules.get("fragment").unwrap(),
			specialization: Specialization::default(),
		};

		let shader_entries = GraphicsShaderSet {
			vertex: vs_entry,
			hull: None,
			domain: None,
			geometry: None,
			fragment: Some(fs_entry),
		};

		let subpass = Subpass {
			index: 0,
			main_pass: render_pass,
		};

		let mut pipeline_desc = GraphicsPipelineDesc::new(
			shader_entries,
			Primitive::TriangleList,
			Rasterizer::FILL,
			&pipeline_layout,
			subpass,
		);

		pipeline_desc
			.blender
			.targets
			.push(ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA));
		pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
			binding: 0,
			stride: std::mem::size_of::<V>() as u32,
			rate: 0,
		});
		pipeline_desc.attributes.push(pso::AttributeDesc {
			location: 0,
			binding: 0,
			element: pso::Element {
				format: Format::Rgb32Float,
				offset: 0,
			},
		});
		pipeline_desc.attributes.push(pso::AttributeDesc {
			location: 1,
			binding: 0,
			element: pso::Element {
				format: Format::Rgba32Float,
				offset: (std::mem::size_of::<f32>() * 3) as u32,
			},
		});

		let pipeline = unsafe {
			device
				.create_graphics_pipeline(&pipeline_desc, None)
				.unwrap()
		};
		
		drop(pipeline_desc);
		
		for (_, shader_module) in shader_modules {
			unsafe { device.destroy_shader_module(shader_module) };
		}
		
		(pipeline, pipeline_layout, desc_sets)
	}

	pub fn build_swapchain(
		device: &B::Device,
		adapter: &mut Adapter<B>,
		surface: &mut B::Surface,
		format: Format,
		old_swapchain: Option<B::Swapchain>,
		width: u32,
		height: u32,
	) -> (B::Swapchain, Backbuffer<B>) {
		let extent = Extent2D { width, height };
		let (caps, _formats, _present_modes, _composite_alphas) =
			surface.compatibility(&mut adapter.physical_device);
		let swap_config = SwapchainConfig::from_caps(&caps, format, extent);

		unsafe { device.create_swapchain(surface, swap_config, old_swapchain).unwrap() }
	}

	pub fn recreate_swapchain(&mut self) {
		debug!("Recreating swapchain");
		self.device.wait_idle().unwrap();
		self.update_dims();
		let dims = self.get_dims();
		let width = dims.0;
		let height = dims.1;
		
		// Clean up resources
		while let Some(image) = self.swapchain_images.pop() {
			//unsafe { self.device.destroy_image(image) }
		}
		while let Some(frame_view) = self.frame_views.pop() {
			unsafe { self.device.destroy_image_view(frame_view) };
		}
		while let Some(framebuffer) = self.framebuffers.pop() {
			unsafe { self.device.destroy_framebuffer(framebuffer) };
		}
		
		let old_swapchain = std::mem::replace(&mut self.swapchain, unsafe { std::mem::uninitialized() });
		//unsafe { self.device.destroy_swapchain(old_swapchain) };
		let (swapchain, backbuffer) = Self::build_swapchain(
			&self.device,
			&mut self.adapter,
			&mut self.surface,
			self.format,
			Some(old_swapchain),
			width,
			height,
		);
		let (swapchain_images, frame_views, framebuffers) = Self::get_frame_views_and_buffers(
			&self.device,
			&self.render_pass,
			backbuffer,
			width,
			height,
		);
		
		// Put in new resources
		unsafe { std::ptr::write(&mut self.swapchain, swapchain); }
		self.swapchain_images = swapchain_images;
		self.frame_views = frame_views;
		self.framebuffers = framebuffers;
		self.viewport.rect.w = width as _;
		self.viewport.rect.h = height as _;
		self.recreate_swapchain = false;
	}

	pub fn get_frame_views_and_buffers(
		device: &B::Device,
		render_pass: &B::RenderPass,
		backbuffer: Backbuffer<B>,
		width: u32,
		height: u32,
	) -> (Vec<B::Image>, Vec<B::ImageView>, Vec<B::Framebuffer>) {
		match backbuffer {
			Backbuffer::Images(images) => {
				let extent = Extent {
					width,
					height,
					depth: 1,
				};

				let color_range = SubresourceRange {
					aspects: Aspects::COLOR,
					levels: 0..1,
					layers: 0..1,
				};

				let image_views = images
					.iter()
					.map(|image| unsafe {
						device
							.create_image_view(
								image,
								ViewKind::D2,
								Format::Bgra8Srgb,
								Swizzle::NO,
								color_range.clone(),
							)
							.unwrap()
					})
					.collect::<Vec<_>>();

				let fbos = image_views
					.iter()
					.map(|image_view| unsafe {
						device
							.create_framebuffer(render_pass, vec![image_view], extent)
							.unwrap()
					})
					.collect();

				(images, image_views, fbos)
			}
			Backbuffer::Framebuffer(fbo) => (Vec::new(), Vec::new(), vec![fbo]),
		}
	}

	pub fn create_semaphore_and_fence(device: &B::Device) -> (B::Semaphore, B::Semaphore) {
		(
			device.create_semaphore().unwrap(),
			device.create_semaphore().unwrap(),
		)
	}
	
	pub fn create_fence(device: &B::Device) -> B::Fence {
		device.create_fence(false).unwrap()
	}

	pub fn create_vertex_buffer<V: Copy>(&self, data: &[V]) -> (B::Buffer, B::Memory) {
		let buffer_stride = std::mem::size_of::<V>() as u64;
		let buffer_len = data.len() as u64 * buffer_stride;

		let mut vertex_buffer = unsafe {
			self.device
				.create_buffer(buffer_len, buffer::Usage::VERTEX)
				.unwrap()
		};

		let buffer_req = unsafe { self.device.get_buffer_requirements(&vertex_buffer) };

		let memory_types = self
			.adapter
			.physical_device
			.memory_properties()
			.memory_types;
		let upload_type = memory_types
			.iter()
			.enumerate()
			.position(|(id, memory_type)| {
				buffer_req.type_mask & (1 << id) != 0
					&& memory_type
						.properties
						.contains(memory::Properties::CPU_VISIBLE)
			})
			.unwrap()
			.into();

		let buffer_memory = unsafe {
			self.device
				.allocate_memory(upload_type, buffer_req.size)
				.unwrap()
		};
		unsafe {
			self.device
				.bind_buffer_memory(&buffer_memory, 0, &mut vertex_buffer)
				.unwrap()
		};

		unsafe {
			let mut vertices = self
				.device
				.acquire_mapping_writer::<V>(&buffer_memory, 0..buffer_req.size)
				.unwrap();
			vertices[0..data.len()].copy_from_slice(data);
			self.device.release_mapping_writer(vertices).unwrap();
		}

		(vertex_buffer, buffer_memory)
	}
	
	pub fn create_uniform_buffer<U: Copy>(device: &B::Device, adapter: &Adapter<B>) -> (B::Buffer, B::Memory) {
		let buffer_len = std::mem::size_of::<U>() as u64;
		
		let mut uniform_buffer = unsafe {
			device
				.create_buffer(buffer_len, buffer::Usage::UNIFORM)
				.unwrap()
		};
		
		let buffer_req = unsafe { device.get_buffer_requirements(&uniform_buffer) };
		
		let mut mem_properties = memory::Properties::empty();
		mem_properties.insert(memory::Properties::CPU_VISIBLE);
		mem_properties.insert(memory::Properties::COHERENT);
		
		let memory_types = adapter
			.physical_device
			.memory_properties()
			.memory_types;
		let upload_type = memory_types
			.iter()
			.enumerate()
			.position(|(id, memory_type)| {
				buffer_req.type_mask & (1 << id) != 0
					&& memory_type
					.properties
					.contains(mem_properties)
			})
			.unwrap()
			.into();
		
		let buffer_memory = unsafe { device.allocate_memory(upload_type, buffer_req.size).unwrap() };
		
		unsafe {
			device.bind_buffer_memory(&buffer_memory, 0, &mut uniform_buffer)
				.unwrap();
		}
		
		(uniform_buffer, buffer_memory)
	}

	pub fn update_dims(&mut self) {
		let size = self.window.get_inner_size().unwrap().to_physical(1.0);
		self.dims = (size.width as _, size.height as _)
	}
	
	pub fn get_dims(&self) -> (u32, u32) {
		self.dims
	}

	pub fn create_viewport(width: u32, height: u32) -> Viewport {
		Viewport {
			rect: Rect {
				x: 0,
				y: 0,
				w: width as _,
				h: height as _,
			},
			depth: 0.0..1.0,
		}
	}

	pub fn create_cmd_buffer(&mut self) -> CommandBuffer<B, Graphics, command::OneShot> {
		self.command_pool
			.acquire_command_buffer::<command::OneShot>()
	}
	
	pub fn update_mvp(&self, data: td::ModelViewProjection) {
		unsafe {
			let buffer_req = self.device.get_buffer_requirements(&self.mvp_buffer);
			let mut writer = self.device.acquire_mapping_writer::<td::ModelViewProjection>(&self.mvp_memory, 0..buffer_req.size).unwrap();
			writer[0] = data;
			self.device.release_mapping_writer(writer).unwrap();
		}
	}

	pub fn render_vertices(&mut self, vertex_buffer: B::Buffer, mvp: td::ModelViewProjection) {
		if self.recreate_swapchain {
			info!("Recreating swapchain");
			self.recreate_swapchain();
		}
		
		let frame_index: hal::SwapImageIndex = unsafe {
			self.device.reset_fence(&self.frame_fence).unwrap();
			self.command_pool.reset();
			match self
				.swapchain
				.acquire_image(!0, FrameSync::Semaphore(&mut self.frame_semaphore))
			{
				Ok(i) => i,
				Err(e) => {
					warn!("Could not obtain swapchain image index: {:?}", e);
					self.recreate_swapchain = true;
					return;
				}
			}
		};
		trace!("Got frame index");
		
		self.update_mvp(mvp);

		let mut cmd_buffer = self.create_cmd_buffer();
		trace!("Created cmd buf");
		unsafe {
			cmd_buffer.begin();
			cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
			cmd_buffer.set_scissors(0, &[self.viewport.rect]);
			cmd_buffer.bind_graphics_pipeline(&self.pipeline);
			cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));
			cmd_buffer.bind_graphics_descriptor_sets(&self.pipeline_layout, 0, &self.descriptor_sets, &[0]);

			{
				let mut encoder = cmd_buffer.begin_render_pass_inline(
					&self.render_pass,
					&self.framebuffers[frame_index as usize],
					self.viewport.rect,
					&[ClearValue::Color(ClearColor::Float([0.4314, 0.5804, 0.8196, 1.0]))],
				);
				encoder.draw(0..6, 0..1);
			}

			cmd_buffer.finish();
		}
		trace!("Finished cmd buf, creating submission");

		let submission = Submission {
			command_buffers: Some(&cmd_buffer),
			wait_semaphores: Some((&self.frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)),
			signal_semaphores: vec![&self.present_semaphore],
		};

		unsafe {
			trace!("Submitting");
			self.queue_group.queues[0].submit(submission, Some(&self.frame_fence));
			trace!("Freeing cmd buf");
			self.device.wait_for_fence(&self.frame_fence, std::u64::MAX).unwrap();
			self.command_pool.free(Some(cmd_buffer));
			trace!("Presenting");
			if let Err(e) = self
				.swapchain
				.present(&mut self.queue_group.queues[0], frame_index, &[&self.present_semaphore])
			{
				warn!("Could not present: {:?}", e);
				self.recreate_swapchain = true;
			};
			trace!("Done");
		}
	}
}
