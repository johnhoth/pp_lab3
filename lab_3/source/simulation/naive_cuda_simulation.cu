#include "naive_cuda_simulation.cuh"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include "simulation/constants.h"  // для epoch_in_seconds
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_wrappers.cuh"

void NaiveCudaSimulation::allocate_device_memory(Universe& universe, void** d_weights, void** d_forces, void** d_velocities, void** d_positions) {
    // Allocate memory for weights (double array)
    parprog_cudaMalloc(d_weights, universe.num_bodies * sizeof(double));
    
    // Allocate memory for forces, velocities and positions (double2 arrays)
    parprog_cudaMalloc(d_forces, universe.num_bodies * sizeof(double2));
    parprog_cudaMalloc(d_velocities, universe.num_bodies * sizeof(double2));
    parprog_cudaMalloc(d_positions, universe.num_bodies * sizeof(double2));
}

void NaiveCudaSimulation::free_device_memory(void** d_weights, void** d_forces, void** d_velocities, void** d_positions) {
    parprog_cudaFree(*d_weights);
    parprog_cudaFree(*d_forces);
    parprog_cudaFree(*d_velocities);
    parprog_cudaFree(*d_positions);
    
    // Prevent dangling pointers
    *d_weights = nullptr;
    *d_forces = nullptr;
    *d_velocities = nullptr;
    *d_positions = nullptr;
}

void NaiveCudaSimulation::copy_data_to_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions) {
    // Copy weights directly
    parprog_cudaMemcpy(d_weights, universe.weights.data(), universe.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    
    // Convert and copy vectors to double2 arrays
    std::vector<double2> temp_vector(universe.num_bodies);
    
    // Convert and copy forces
    for (size_t i = 0; i < universe.num_bodies; ++i) {
        temp_vector[i].x = universe.forces[i][0];
        temp_vector[i].y = universe.forces[i][1];
    }
    parprog_cudaMemcpy(d_forces, temp_vector.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    
    // Convert and copy velocities
    for (size_t i = 0; i < universe.num_bodies; ++i) {
        temp_vector[i].x = universe.velocities[i][0];
        temp_vector[i].y = universe.velocities[i][1];
    }
    parprog_cudaMemcpy(d_velocities, temp_vector.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    
    // Convert and copy positions
    for (size_t i = 0; i < universe.num_bodies; ++i) {
        temp_vector[i].x = universe.positions[i][0];
        temp_vector[i].y = universe.positions[i][1];
    }
    parprog_cudaMemcpy(d_positions, temp_vector.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
}

void NaiveCudaSimulation::copy_data_from_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions) {
    // Copy weights directly
    parprog_cudaMemcpy(universe.weights.data(), d_weights, universe.num_bodies * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Temporary buffer for double2 data
    std::vector<double2> temp_vector(universe.num_bodies);
    
    // Copy and convert forces
    parprog_cudaMemcpy(temp_vector.data(), d_forces, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < universe.num_bodies; ++i) {
        universe.forces[i][0] = temp_vector[i].x;
        universe.forces[i][1] = temp_vector[i].y;
    }
    
    // Copy and convert velocities
    parprog_cudaMemcpy(temp_vector.data(), d_velocities, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < universe.num_bodies; ++i) {
        universe.velocities[i][0] = temp_vector[i].x;
        universe.velocities[i][1] = temp_vector[i].y;
    }
    
    // Copy and convert positions
    parprog_cudaMemcpy(temp_vector.data(), d_positions, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < universe.num_bodies; ++i) {
        universe.positions[i][0] = temp_vector[i].x;
        universe.positions[i][1] = temp_vector[i].y;
    }
}

__global__
void calculate_forces_kernel(std::uint32_t num_bodies, double2* d_positions, double* d_weights, double2* d_forces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    double2 total_force = make_double2(0.0, 0.0);
    double2 pos_i = d_positions[i];
    double weight_i = d_weights[i];

    // Calculate forces from all other bodies
    for (uint32_t j = 0; j < num_bodies; j++) {
        if (i == j) continue;

        double2 pos_j = d_positions[j];
        double weight_j = d_weights[j];

        // Calculate distance vector and magnitude
        double2 dist_vec = make_double2(pos_j.x - pos_i.x, pos_j.y - pos_i.y);
        double dist_squared = dist_vec.x * dist_vec.x + dist_vec.y * dist_vec.y;
        
        // Calculate force magnitude using gravitational formula
        double force_magnitude = G * weight_i * weight_j / dist_squared;
        
        // Calculate force direction
        double dist = sqrt(dist_squared);
        double2 force = make_double2(
            force_magnitude * dist_vec.x / dist,
            force_magnitude * dist_vec.y / dist
        );

        // Add to total force
        total_force.x += force.x;
        total_force.y += force.y;
    }

    // Store result
    d_forces[i] = total_force;
}

void NaiveCudaSimulation::calculate_forces(Universe& universe, void* d_positions, void* d_weights, void* d_forces) {
    // Configure grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (universe.num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    calculate_forces_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        universe.num_bodies,
        (double2*)d_positions,
        (double*)d_weights,
        (double2*)d_forces
    );
}

__global__
void calculate_velocities_kernel(std::uint32_t num_bodies, double2* d_forces, double* d_weights, double2* d_velocities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    // Get force and weight for body i
    double2 force = d_forces[i];
    double weight = d_weights[i];
    double2 velocity = d_velocities[i];

    // Calculate acceleration: a = F/m
    double2 acceleration = make_double2(
        force.x / weight,
        force.y / weight
    );

    // Update velocity: v = v0 + a*t
    velocity.x += acceleration.x * epoch_in_seconds;
    velocity.y += acceleration.y * epoch_in_seconds;

    // Store result
    d_velocities[i] = velocity;
}

void NaiveCudaSimulation::calculate_velocities(Universe& universe, void* d_forces, void* d_weights, void* d_velocities) {
    // Configure grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (universe.num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    calculate_velocities_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        universe.num_bodies,
        (double2*)d_forces,
        (double*)d_weights,
        (double2*)d_velocities
    );
}

__global__
void calculate_positions_kernel(std::uint32_t num_bodies, double2* d_velocities, double2* d_positions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    // Get velocity and current position
    double2 velocity = d_velocities[i];
    double2 position = d_positions[i];

    // Calculate displacement: s = v * t
    double2 displacement = make_double2(
        velocity.x * epoch_in_seconds,
        velocity.y * epoch_in_seconds
    );

    // Update position: p = p0 + s
    position.x += displacement.x;
    position.y += displacement.y;

    // Store result
    d_positions[i] = position;
}

void NaiveCudaSimulation::calculate_positions(Universe& universe, void* d_velocities, void* d_positions) {
    // Configure grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (universe.num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    calculate_positions_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        universe.num_bodies,
        (double2*)d_velocities,
        (double2*)d_positions
    );
}

void NaiveCudaSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    // Pointers for device memory
    void *d_weights = nullptr;
    void *d_forces = nullptr;
    void *d_velocities = nullptr;
    void *d_positions = nullptr;

    // Allocate device memory
    allocate_device_memory(universe, &d_weights, &d_forces, &d_velocities, &d_positions);

    // Copy initial state to device
    copy_data_to_device(universe, d_weights, d_forces, d_velocities, d_positions);

    // Run simulation for specified number of epochs
    for (std::uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs,
                      d_weights, d_forces, d_velocities, d_positions);
    }

    // Copy final state back to host
    copy_data_from_device(universe, d_weights, d_forces, d_velocities, d_positions);

    // Free device memory
    free_device_memory(&d_weights, &d_forces, &d_velocities, &d_positions);
}

__global__
void get_pixels_kernel(std::uint32_t num_bodies, double2* d_positions, std::uint8_t* d_pixels, 
                      std::uint32_t plot_width, std::uint32_t plot_height,
                      double plot_bounding_box_x_min, double plot_bounding_box_x_max,
                      double plot_bounding_box_y_min, double plot_bounding_box_y_max) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= plot_width * plot_height) return;

    // Calculate pixel coordinates in plot space
    // Convert from bottom-left to top-left indexing
    int pixel_y = (plot_height - 1) - (pixel_idx / plot_width);
    int pixel_x = pixel_idx % plot_width;

    // Calculate pixel bounds in universe space
    double pixel_width = (plot_bounding_box_x_max - plot_bounding_box_x_min) / (plot_width - 1);
    double pixel_height = (plot_bounding_box_y_max - plot_bounding_box_y_min) / (plot_height - 1);

    double pixel_min_x = plot_bounding_box_x_min + pixel_x * pixel_width - pixel_width/2;
    double pixel_max_x = pixel_min_x + pixel_width;
    double pixel_min_y = plot_bounding_box_y_min + pixel_y * pixel_height - pixel_height/2;
    double pixel_max_y = pixel_min_y + pixel_height;

    // Check if any body lies within pixel bounds
    bool pixel_active = false;
    for (uint32_t i = 0; i < num_bodies && !pixel_active; ++i) {
        double2 pos = d_positions[i];
        if (pos.x >= pixel_min_x && pos.x < pixel_max_x &&
            pos.y >= pixel_min_y && pos.y < pixel_max_y) {
            pixel_active = true;
        }
    }

    d_pixels[pixel_idx] = pixel_active ? 1 : 0;
}

std::vector<std::uint8_t> NaiveCudaSimulation::get_pixels(std::uint32_t plot_width, std::uint32_t plot_height,
                                                         BoundingBox plot_bounding_box, void* d_positions,
                                                         std::uint32_t num_bodies) {
    // Prepare host pixels vector
    std::vector<std::uint8_t> pixels(plot_width * plot_height);

    // Allocate device memory for pixels
    void* d_pixels;  // Изменен тип с uint8_t* на void*
    parprog_cudaMalloc(&d_pixels, plot_width * plot_height * sizeof(std::uint8_t));

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (plot_width * plot_height + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    get_pixels_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        num_bodies,
        (double2*)d_positions,
        (std::uint8_t*)d_pixels,  // Приведение типа
        plot_width,
        plot_height,
        plot_bounding_box.x_min,
        plot_bounding_box.x_max,
        plot_bounding_box.y_min,
        plot_bounding_box.y_max
    );

    // Copy result back to host
    parprog_cudaMemcpy(pixels.data(), d_pixels, plot_width * plot_height * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    parprog_cudaFree(d_pixels);

    return pixels;
}

__global__
void compress_pixels_kernel(std::uint32_t num_raw_pixels, std::uint8_t* d_raw_pixels, std::uint8_t* d_compressed_pixels) {
    int compressed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int raw_idx = compressed_idx * 8;
    
    if (raw_idx >= num_raw_pixels) return;

    // Each thread processes 8 pixels and compresses them into one byte
    std::uint8_t compressed_byte = 0;
    
    // Set bits from LSB to MSB (raw_pixels[0] -> bit 0, raw_pixels[7] -> bit 7)
    for (int i = 0; i < 8; i++) {
        if (d_raw_pixels[raw_idx + i] != 0) {
            compressed_byte |= (1 << i);
        }
    }
    
    d_compressed_pixels[compressed_idx] = compressed_byte;
}

void NaiveCudaSimulation::compress_pixels(std::vector<std::uint8_t>& raw_pixels, std::vector<std::uint8_t>& compressed_pixels) {
    std::uint32_t num_raw_pixels = raw_pixels.size();
    std::uint32_t num_compressed_pixels = num_raw_pixels / 8;
    
    // Allocate device memory
    void* d_raw_pixels;
    void* d_compressed_pixels;
    parprog_cudaMalloc(&d_raw_pixels, num_raw_pixels * sizeof(std::uint8_t));
    parprog_cudaMalloc(&d_compressed_pixels, num_compressed_pixels * sizeof(std::uint8_t));
    
    // Copy raw pixels to device
    parprog_cudaMemcpy(d_raw_pixels, raw_pixels.data(), num_raw_pixels * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_compressed_pixels + threadsPerBlock - 1) / threadsPerBlock;
    compress_pixels_kernel<<<blocksPerGrid, threadsPerBlock>>>(num_raw_pixels, (std::uint8_t*)d_raw_pixels, (std::uint8_t*)d_compressed_pixels);
    
    // Copy result back to host
    parprog_cudaMemcpy(compressed_pixels.data(), d_compressed_pixels, num_compressed_pixels * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    
    // Free device memory
    parprog_cudaFree(d_raw_pixels);
    parprog_cudaFree(d_compressed_pixels);
}

void NaiveCudaSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs, void* d_weights, void* d_forces, void* d_velocities, void* d_positions) {
    calculate_forces(universe, d_positions, d_weights, d_forces);
    calculate_velocities(universe, d_forces, d_weights, d_velocities);
    calculate_positions(universe, d_velocities, d_positions);

    universe.current_simulation_epoch++;
    if(create_intermediate_plots) {
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0) {
            std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), 
                                                        plotter.get_plot_bounding_box(), d_positions, universe.num_bodies);
            
            // Use compression for plotting
            std::vector<std::uint8_t> compressed_pixels;
            compressed_pixels.resize(pixels.size()/8);
            compress_pixels(pixels, compressed_pixels);
            plotter.add_compressed_pixels_to_image(compressed_pixels);

            plotter.write_and_clear();
        }
    }
}

void NaiveCudaSimulation::calculate_forces_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_positions, void* d_weights, void* d_forces){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
}

void NaiveCudaSimulation::calculate_velocities_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_forces, void* d_weights, void* d_velocities){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_velocities_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_forces, (double*) d_weights, (double2*) d_velocities);
}

void NaiveCudaSimulation::calculate_positions_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_velocities, void* d_positions){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_velocities, (double2*) d_positions);
}
