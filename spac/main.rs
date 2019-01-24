use std::{collections::HashMap, io::Read};

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
use winit::{DeviceEvent, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

pub mod td;

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
	let (vertex_buffer, vertex_memory) = renderer.create_vertex_buffer(&td::TRI);
	debug!("Allocated memory, starting main loop");

	let mut running = true;

	while running {
		debug!("loop, polling events");
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
					}
					_ => {}
				},
				_ => {}
			}
		});
		debug!("rendering vertices");
		renderer.render_vertices(vertex_buffer);
		std::thread::sleep(std::time::Duration::from_millis(16));
	}
}

pub struct Renderer<B, I>
where
	B: hal::Backend,
	I: hal::Instance<Backend = B>,
{
	window: Window,
	instance: I,
	surface: B::Surface,
	adapter: Adapter<B>,
	device: B::Device,
	queue_group: QueueGroup<B, Graphics>,
	command_pool: CommandPool<B, Graphics>,
	format: Format,
	render_pass: B::RenderPass,
	pipeline: B::GraphicsPipeline,
	swapchain: B::Swapchain,
	frame_views: Vec<B::ImageView>,
	framebuffers: Vec<B::Framebuffer>,
	semaphore: B::Semaphore,
	fence: B::Fence,
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
		let (device, mut queue_group) = Self::get_device_and_queue_group(&adapter, &surface);
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
		let vert_spirv: &[u8] = include_bytes!("../assets/vert.spv");
		let frag_spirv: &[u8] = include_bytes!("../assets/frag.spv");
		let shaders = Self::load_spirv_shaders(
			&device,
			vec![
				(String::from("vertex"), vert_spirv),
				(String::from("fragment"), frag_spirv),
			],
		);
		let pipeline = Self::build_pipeline::<td::Vertex>(&device, &render_pass, &shaders);
		let (swapchain, backbuffer) =
			Self::build_swapchain(&device, &mut adapter, &mut surface, format, WIDTH, HEIGHT);
		let (frame_views, framebuffers) =
			Self::get_frame_views_and_buffers(&device, &render_pass, backbuffer, WIDTH, HEIGHT);
		let (semaphore, fence) = Self::create_semaphore_and_fence(&device);
		let viewport = Self::create_viewport(WIDTH, HEIGHT);
		let recreate_swapchain = false;
		let recreate_swapchain_dims = None;

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
			swapchain,
			frame_views,
			framebuffers,
			semaphore,
			fence,
			viewport,
			recreate_swapchain,
			recreate_swapchain_dims,
		}
	}

	pub fn create_window(width: u32, height: u32) -> (Window, EventsLoop) {
		unimplemented!()
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

	pub fn build_pipeline<V>(
		device: &B::Device,
		render_pass: &B::RenderPass,
		shader_modules: &HashMap<String, B::ShaderModule>,
	) -> B::GraphicsPipeline {
		let pipeline_layout = unsafe { device.create_pipeline_layout(&[], &[]).unwrap() };

		let vs_entry = EntryPoint::<B> {
			entry: "main",
			module: shader_modules.get("vertex").unwrap(),
			specialization: Specialization {
				constants: &[SpecializationConstant { id: 0, range: 0..4 }],
				data: unsafe { std::mem::transmute::<&f32, &[u8; 4]>(&0.8f32) },
			},
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
				offset: 12,
			},
		});

		unsafe {
			device
				.create_graphics_pipeline(&pipeline_desc, None)
				.unwrap()
		}
	}

	pub fn build_swapchain(
		device: &B::Device,
		adapter: &mut Adapter<B>,
		surface: &mut B::Surface,
		format: Format,
		width: u32,
		height: u32,
	) -> (B::Swapchain, Backbuffer<B>) {
		let extent = Extent2D { width, height };
		let (caps, _formats, _present_modes, _composite_alphas) =
			surface.compatibility(&mut adapter.physical_device);
		let swap_config = SwapchainConfig::from_caps(&caps, format, extent);

		unsafe { device.create_swapchain(surface, swap_config, None).unwrap() }
	}

	pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
		debug!("Recreated swapchain");
		self.device.wait_idle().unwrap();
		let (swapchain, backbuffer) = Self::build_swapchain(
			&self.device,
			&mut self.adapter,
			&mut self.surface,
			self.format,
			width,
			height,
		);
		let (frame_views, framebuffers) = Self::get_frame_views_and_buffers(
			&self.device,
			&self.render_pass,
			backbuffer,
			width,
			height,
		);
		self.swapchain = swapchain;
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
	) -> (Vec<B::ImageView>, Vec<B::Framebuffer>) {
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

				(image_views, fbos)
			}
			Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
		}
	}

	pub fn create_semaphore_and_fence(device: &B::Device) -> (B::Semaphore, B::Fence) {
		(
			device.create_semaphore().unwrap(),
			device.create_fence(false).unwrap(),
		)
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

	pub fn get_dims(&self) -> (u32, u32) {
		let size = self.window.get_inner_size().unwrap().to_physical(1.0);
		(size.width as _, size.height as _)
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

	pub fn render_vertices(&mut self, vertex_buffer: B::Buffer) {
		if let Some(dims) = self.recreate_swapchain_dims.take() {
			self.recreate_swapchain(dims.0, dims.1);
		}

		let frame_index: hal::SwapImageIndex = unsafe {
			self.device.reset_fence(&self.fence).unwrap();
			self.command_pool.reset();
			match self
				.swapchain
				.acquire_image(!0, FrameSync::Semaphore(&mut self.semaphore))
			{
				Ok(i) => i,
				Err(_) => {
					let dims = self.get_dims();
					self.recreate_swapchain(dims.0, dims.1);
					self.render_vertices(vertex_buffer);
					return;
				}
			}
		};
		debug!("Got frame index");

		let mut cmd_buffer = self.create_cmd_buffer();
		debug!("Created cmd buf");
		unsafe {
			cmd_buffer.begin();
			cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
			cmd_buffer.set_scissors(0, &[self.viewport.rect]);
			cmd_buffer.bind_graphics_pipeline(&self.pipeline);
			cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));

			{
				let mut encoder = cmd_buffer.begin_render_pass_inline(
					&self.render_pass,
					&self.framebuffers[frame_index as usize],
					self.viewport.rect,
					&[ClearValue::Color(ClearColor::Float([1.0, 1.0, 1.0, 1.0]))],
				);
				encoder.draw(0..6, 0..1);
			}

			cmd_buffer.finish();
		}
		debug!("Finished cmd buf, creating submission");

		let submission = Submission {
			command_buffers: Some(&cmd_buffer),
			wait_semaphores: Some((&self.semaphore, PipelineStage::BOTTOM_OF_PIPE)),
			signal_semaphores: &[],
		};

		unsafe {
			debug!("Submitting");
			self.queue_group.queues[0].submit(submission, Some(&mut self.fence));
			debug!("Waiting");
			self.device.wait_for_fence(&self.fence, !0).unwrap();
			debug!("Freeing cmd buf");
			self.command_pool.free(Some(cmd_buffer));
			debug!("Presenting");
			if let Err(_) = self
				.swapchain
				.present_nosemaphores(&mut self.queue_group.queues[0], frame_index)
			{
				let dims = self.get_dims();
				self.recreate_swapchain(dims.0, dims.1);
			};
			debug!("Done");
		}
	}
}
