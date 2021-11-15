use rand::{Rng, thread_rng};

struct SimulationState {
    start_time: std::time::Instant,
    elapsed_generation: u64,
    current_generation_step: u64,
    simulation_board: SimulationBoard,
    simulation_config: &'static SimulationConfig,
}

impl SimulationState {
    fn new(simulation_config: &'static SimulationConfig) -> Self {
        let simulation_board: SimulationBoard = SimulationBoard::new(simulation_config.board_config.height, simulation_config.board_config.width);
        Self {
            start_time: std::time::Instant::now(),
            elapsed_generation: 0,
            current_generation_step: 0,
            simulation_board: simulation_board,
            simulation_config,
        }
    }

    fn run(&mut self) {
        let mut first_run = true;
        while self.elapsed_generation < self.simulation_config.max_generations as u64 || self.simulation_config.max_generations == 0 {
            println!("{:?}: Generation: {}", std::time::Instant::now() - self.start_time, self.elapsed_generation);
            self.run_generation(first_run);
            first_run = false;
        }
    }

    fn run_generation(&mut self, first_run: bool) {
        if first_run {
            self.simulation_board.populate(self.simulation_config.initial_population, self.simulation_config);
        }
        println!("Alive organisms before generation count: {:?}", self.simulation_board.alive_organisms_count());
        if !first_run {
            self.simulation_board.reproduce_organisms(self.simulation_config);
        }
        while self.current_generation_step < self.simulation_config.steps_per_generation as u64 {
            self.step();
            self.current_generation_step += 1;
        }
        self.select_organisms();
        self.elapsed_generation += 1;
    }

    fn select_organisms(&mut self) {
        let organisms = self.simulation_board.cells.iter().flat_map(|row| row.iter()).filter(|cell| cell.organism.is_some()).map(|cell| cell.organism.as_ref().unwrap().clone()).collect::<Vec<_>>();
        let alive_organisms_count = organisms.len();
        for og in organisms {
            match (self.simulation_config.selection_method)(self, &og) {
                true => {
                    self.simulation_board.cells[og.position.x as usize][og.position.y as usize].organism = Some(og.clone());
                },
                false => {
                    self.simulation_board.cells[og.position.x as usize][og.position.y as usize].organism = None;
                }
            }
        }
        println!("Alive organisms after selection count:         {:?}", self.simulation_board.alive_organisms_count());
        println!("Dead organisms after selection count:          {:?}", alive_organisms_count - self.simulation_board.alive_organisms_count());
        println!("Percentage of organisms alive after selection: {:?}%", self.simulation_board.alive_organisms_count() as f64 / alive_organisms_count as f64 * 100.0);
    }

    fn step(&mut self) {
        for row in self.simulation_board.cells.clone().iter() {
            for cell in row.iter() {
                if cell.organism.is_some() {
                    let organism = cell.organism.as_ref().unwrap();

                    for n in organism.brain.neural_net.neurons.iter() {
                        println!("{:?}", n.neuron_type.primitive_type);
                        if n.neuron_type.primitive_type == PrimitiveNeuronType::Output { 
                            let inputs = self.neuron_collect_inputs(organism, n.neuron_type);
                            (self
                                .simulation_config
                                .brain_config
                                .neuron_types[n.neuron_type.as_index(self.simulation_config)]
                                .post_activation_system.unwrap()
                            )(
                                self,
                                organism,
                                (self
                                    .simulation_config
                                    .brain_config
                                    .neuron_types[n.neuron_type.as_index(self.simulation_config)]
                                    .inner_system
                                )(
                                    self,
                                    organism,
                                    inputs
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    fn neuron_collect_inputs(&self, organism: &Organism, neuron_type: NeuronIdentifier) -> Vec<f32> {
        let mut inputs: Vec<f32> = Vec::new();
        for synapse in &organism.brain.neural_net.neurons[neuron_type.as_index(self.simulation_config)].owned_synapses { // Error: Index out of bound, debug the as_index function from neuron identifier
            inputs.push(
                (self.simulation_config.brain_config.neuron_types[neuron_type.as_index(self.simulation_config)].inner_system)(
                    self,
                    organism,
                    self.neuron_collect_inputs(organism, synapse.origin)
                )
            );
        }
        // TODO: Implement post activation function
        inputs
    }
}

type UniqueNeuronTypeArray = &'static[NeuronType];
struct BrainConfig {
    unique_input_neurons_count: usize,
    unique_output_neurons_count: usize,
    unique_hidden_neurons_count: usize,
    synapses: usize,
    neuron_types: UniqueNeuronTypeArray,
}

struct BoardConfig {
    width: usize,
    height: usize,
}

struct EvolutionConfig {
    reproduction_factor: i8,
    mutation_probability: f32,
}
type SelectionMethod = fn(&mut SimulationState, organism: &Organism) -> bool;
struct SimulationConfig {
    brain_config: BrainConfig,
    board_config: BoardConfig,
    evolution_config: EvolutionConfig,
    initial_population: usize,
    steps_per_second: usize,
    max_generations: usize,
    steps_per_generation: usize,
    selection_method: SelectionMethod,
}

type Gen = Synapse;
impl Gen {
    fn new_random(simulation_config: &SimulationConfig) -> Self {
        let mut rng = thread_rng();
        Synapse {
            origin: NeuronIdentifier::random_inputable(simulation_config),
            target: NeuronIdentifier::random_outputable(simulation_config),
            weight: rng.gen_range(-4.0..4.0), // [-4.0, 4.0]
        }
    }
}
#[derive(Debug, Clone)]
struct Genome {
    gens: Vec<Gen>,
}
impl Genome {
    fn new_random(size: usize, simulation_config: &SimulationConfig) -> Self {
        let mut gens: Vec<Gen> = Vec::with_capacity(size);
        for _ in 0..size {
            gens.push(Gen::new_random(simulation_config));
        }
        Self {
            gens,
        }
    }

    fn new_mutation(genome: &Genome, simulation_config: &SimulationConfig) -> Self {
        let mut rng = thread_rng();
        let mut new_genome = Genome {
            gens: Vec::with_capacity(genome.gens.len()),
        };
        for gen in new_genome.gens.iter_mut() {
            if rng.gen_bool(simulation_config.evolution_config.mutation_probability as f64) {
                match rng.gen_range(0..3) {
                    // TODO: Change for linear mutation for neurons
                    0 => gen.origin = NeuronIdentifier::random_inputable(simulation_config),
                    1 => gen.target = NeuronIdentifier::random_outputable(simulation_config),
                    2 => gen.weight += rng.gen_range(-0.1..0.1),
                    _ => panic!("Invalid mutation type"),
                }
            }
        }
        new_genome
    }
}

#[derive(Debug, Clone)]
enum Orientation {
    North,
    East,
    South,
    West,
}

#[derive(Debug, Clone)]
struct Position {
    x: i32,
    y: i32,
    orientation: Orientation,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum PrimitiveNeuronType {
    Input,
    Hidden,
    Output,
}
type InnerSystem = fn(&SimulationState, &Organism, Vec<f32>) -> f32;
type PostActivationSystem = fn(&mut SimulationState, &Organism, f32);
struct NeuronType {
    primitive_type: PrimitiveNeuronType,
    identifier: NeuronIdentifier,
    inner_system: InnerSystem,
    post_activation_system: std::option::Option<PostActivationSystem>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NeuronIdentifier {
    primitive_type: PrimitiveNeuronType,
    index: usize,
}
impl NeuronIdentifier {
    fn as_index(&self, simulation_config: &SimulationConfig) -> usize {
        match self.primitive_type {
            PrimitiveNeuronType::Input => self.index,
            PrimitiveNeuronType::Hidden => self.index + simulation_config.brain_config.unique_input_neurons_count,
            PrimitiveNeuronType::Output => self.index + simulation_config.brain_config.unique_input_neurons_count + simulation_config.brain_config.unique_hidden_neurons_count,
        }
    }

    fn random_inputable(simulation_config: &SimulationConfig) -> Self {
        let mut rng = thread_rng();
        let sided = rng.gen_bool(0.5);
        let primitive_type = match sided {
            true => PrimitiveNeuronType::Input,
            false => PrimitiveNeuronType::Hidden,
        };
        let index_range = match sided {
            true => 0..simulation_config.brain_config.unique_input_neurons_count,
            false => simulation_config.brain_config.unique_input_neurons_count - 1..simulation_config.brain_config.unique_input_neurons_count + simulation_config.brain_config.unique_hidden_neurons_count - 1
        };
        Self {
            primitive_type,
            index: thread_rng().gen_range(index_range),
        }
    }

    fn random_outputable(simulation_config: &SimulationConfig) -> Self {
        let mut rng = thread_rng();
        let sided = rng.gen_bool(0.5);
        let primitive_type = match sided {
            true => PrimitiveNeuronType::Output,
            false => PrimitiveNeuronType::Hidden,
        };
        let indexRange = match sided {
            true => simulation_config.brain_config.unique_input_neurons_count - 1..simulation_config.brain_config.unique_input_neurons_count + simulation_config.brain_config.unique_hidden_neurons_count - 1,
            false => simulation_config.brain_config.unique_input_neurons_count + simulation_config.brain_config.unique_output_neurons_count - 2..simulation_config.brain_config.unique_input_neurons_count + simulation_config.brain_config.unique_hidden_neurons_count + simulation_config.brain_config.unique_output_neurons_count - 2,
        };
        Self {
            primitive_type,
            index: thread_rng().gen_range(indexRange),
        }
    }
}
    
#[derive(Debug, Clone)]
struct Neuron {
    neuron_type: NeuronIdentifier,
    value: f32,
    owned_synapses: Vec<Synapse>,
}

#[derive(Clone, Copy, Debug)]
struct Synapse {
    origin: NeuronIdentifier,
    target: NeuronIdentifier,
    weight: f64
}

#[derive(Debug, Clone)]
struct NeuralNet {
    neurons: Vec<Neuron>,
}
impl NeuralNet {
    fn new(genome: &Genome, simulation_config: &SimulationConfig) -> Self {
        let synapses = &genome.gens; // Retrieve the synapses from the genome
        let origin_neurons = synapses.iter().map(|synapse| synapse.origin).collect::<Vec<_>>();
        let target_neurons = synapses.iter().map(|synapse| synapse.target).collect::<Vec<_>>();
        let mut used_neurons: Vec<NeuronIdentifier> = Vec::new();
        for n in origin_neurons {
            if simulation_config.brain_config.neuron_types[n.as_index(simulation_config)].primitive_type == PrimitiveNeuronType::Hidden && !target_neurons.contains(&n) {
                continue;
            }
            used_neurons.push(n);
        }
        for n in target_neurons {
            if(used_neurons.contains(&n)) {
                continue;
            }
            used_neurons.push(n);
        }
        let mut used_synapses: Vec<Synapse> = Vec::new();
        for s in synapses {
            if !used_neurons.contains(&s.origin) || !used_neurons.contains(&s.target) {
                continue;
            }
            used_synapses.push(*s);
        }

        Self {
            neurons: used_neurons.iter().map(|n| Neuron {
                neuron_type: *n,
                value: 0.0,
                owned_synapses: used_synapses.iter().cloned().filter(|s| s.target == *n).collect::<Vec<_>>(),
            }).collect(),
        }
    }
}
#[derive(Debug, Clone)]
struct Brain {
    neural_net: NeuralNet,
}
impl Brain {
    fn new(genome: &Genome, simulation_config: &SimulationConfig) -> Self {
        Self {
            neural_net: NeuralNet::new(genome, simulation_config),
        }
    }
}

#[derive(Debug, Clone)]
struct Organism {
    position: Position,
    genome: Genome,
    brain: Brain,
}
impl Organism {
    fn new(genome: Genome, x: i32, y: i32, orientation: Orientation, simulation_config: &SimulationConfig) -> Self {
        Self {
            position: Position {
                x,
                y,
                orientation,
            },
            brain: Brain::new(&genome, simulation_config),
            genome,
        }
    }
    
}

#[derive(Debug, Clone)]
struct Tile {
    organism: Option<Organism>,
}
struct SimulationBoard {
    cells: Vec<Vec<Tile>>,
}

impl SimulationBoard {
    fn new(height: usize, width: usize) -> Self {
        let mut cells = Vec::new();
        for _ in 0..height {
            let mut row = Vec::new();
            for _ in 0..width {
                row.push(Tile { organism: None });
            }
            cells.push(row);
        }
        Self { cells }
    }

    fn clear(&mut self) {
        for row in self.cells.iter_mut() {
            for cell in row.iter_mut() {
                cell.organism = None;
            }
        }
    }

    fn populate(&mut self, population_to_generate: usize, simulation_config: &SimulationConfig) {        
        let mut rng = rand::thread_rng();
        for _ in 0..population_to_generate {
            loop {
                let x = rng.gen_range(0..self.cells.len());
                let y = rng.gen_range(0..self.cells[0].len());
                if self.cells[x][y].organism.is_none() {
                    self.cells[x][y].organism = Some(Organism::new(Genome::new_random(simulation_config.brain_config.synapses, simulation_config), x as i32, y as i32, Orientation::North, simulation_config));
                    break;
                }
            }
        }
    }

    fn reproduce_organisms(&mut self, simulation_config: &SimulationConfig) {
        let mut rng = rand::thread_rng();
        let mut organisms_to_reproduce = self.cells.iter().flat_map(|row| row.iter()).filter(|cell| cell.organism.is_some()).map(|cell| cell.organism.as_ref().unwrap().clone()).collect::<Vec<_>>();
        self.clear();
        let mut organism_count = 0;
        while organism_count < simulation_config.initial_population {
            for og in organisms_to_reproduce.iter_mut() {
                if organism_count > simulation_config.initial_population { //TODO: Better population regulation
                    break;
                }
                loop {
                    let x = rng.gen_range(0..self.cells.len());
                    let y = rng.gen_range(0..self.cells[0].len());
                    if self.cells[x][y].organism.is_none() {
                        self.cells[x][y].organism = Some(Organism::new(Genome::new_mutation(&og.genome, simulation_config), x as i32, y as i32, Orientation::North, simulation_config));
                        break;
                    }
                    organism_count += 1;
                }
            }
        }
    }

    fn alive_organisms_count(&self) -> usize {
        let mut count = 0;
        for row in self.cells.iter() {
            for cell in row.iter() {
                if cell.organism.is_some() {
                    count += 1;
                }
            }
        }
        count
    }
}

const MERGE_INNER_SYSTEM: InnerSystem = |_, _, inputs| {
    let mut sum: f32 = 0.0;
    for input in inputs {
        sum += input;
    }
    println!("{:?}", sum);
    sum.tanh()
};

const NEURON_TYPES_ARRAY: UniqueNeuronTypeArray = &[
    NeuronType {
        // X Position Neuron
        primitive_type: PrimitiveNeuronType::Input,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Input,
            index: 0,
        },
        inner_system: |_, organism, _| {
            (organism.position.x + 1) as f32 / SIMULATION_CONFIG.board_config.width as f32
        },
        post_activation_system: None,
    },
    NeuronType {
        // Y Position Neuron
        primitive_type: PrimitiveNeuronType::Input,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Input,
            index: 1,
        },
        inner_system: |_, organism, _| {
            (organism.position.y + 1) as f32 / SIMULATION_CONFIG.board_config.height as f32
        },
        post_activation_system: None,
    },
    NeuronType {
        // Population density in front of organism
        primitive_type: PrimitiveNeuronType::Input,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Input,
            index: 2,
        },
        inner_system: |sim, organism, _| {
            const RANGE: std::ops::Range<usize> = 0..3;
            let mut sum: f32 = 0.0;
            match organism.position.orientation {
                Orientation::North => {
                    for y in RANGE {
                        sum += match sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize - y].organism.is_some() {
                            true => 1.0,
                            false => 0.0,
                        };
                    }
                },
                Orientation::East => {
                    for x in RANGE {
                        sum += match sim.simulation_board.cells[organism.position.x as usize + x][organism.position.y as usize].organism.is_some() {
                            true => 1.0,
                            false => 0.0,
                        };
                    }
                },
                Orientation::South => {
                    for y in RANGE {
                        sum += match sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize + y].organism.is_some() {
                            true => 1.0,
                            false => 0.0,
                        };
                    }
                },
                Orientation::West => {
                    for x in RANGE {
                        sum += match sim.simulation_board.cells[organism.position.x as usize - x][organism.position.y as usize].organism.is_some() {
                            true => 1.0,
                            false => 0.0,
                        };
                    }
                }
            }
            sum / RANGE.len() as f32
        },
        post_activation_system: None,
    },
    NeuronType {
        primitive_type: PrimitiveNeuronType::Hidden,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Hidden,
            index: 0,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: None,
    },
    NeuronType {
        primitive_type: PrimitiveNeuronType::Hidden,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Hidden,
            index: 1,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: None,
    },
    NeuronType {
        primitive_type: PrimitiveNeuronType::Hidden,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Hidden,
            index: 2,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: None,
    },
    NeuronType {
        // Move North
        primitive_type: PrimitiveNeuronType::Output,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Output,
            index: 0,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: Some(|sim, organism, value| {
            // North
            if value > 0.5 && sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize + 1].organism.is_none() {
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize + 1].organism = Some(organism.clone());
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize + 1].organism.as_mut().unwrap().position.x -= 1;
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize].organism = None;
            }
        }),
    },
    NeuronType {
        // Move East
        primitive_type: PrimitiveNeuronType::Output,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Output,
            index: 1,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: Some(|sim, organism, value| {
            // East
            if value > 0.5 && sim.simulation_board.cells[organism.position.x as usize + 1][organism.position.y as usize].organism.is_none() {
                sim.simulation_board.cells[organism.position.x as usize + 1][organism.position.y as usize].organism = Some(organism.clone());
                sim.simulation_board.cells[organism.position.x as usize + 1][organism.position.y as usize].organism.as_mut().unwrap().position.x -= 1;
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize].organism = None;
            }
        }),
    },
    NeuronType {
        // Move South
        primitive_type: PrimitiveNeuronType::Output,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Output,
            index: 2,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: Some(|sim, organism, value| {
            // South
            if value > 0.5 && sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize - 1].organism.is_none() {
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize - 1].organism = Some(organism.clone());
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize - 1].organism.as_mut().unwrap().position.x -= 1;
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize].organism = None;
            }
        }),
    },
    NeuronType {
        // Move West
        primitive_type: PrimitiveNeuronType::Output,
        identifier: NeuronIdentifier {
            primitive_type: PrimitiveNeuronType::Output,
            index: 3,
        },
        inner_system: MERGE_INNER_SYSTEM,
        post_activation_system: Some(|sim, organism, value| {
            // West
            if value > 0.5 && sim.simulation_board.cells[organism.position.x as usize - 1][organism.position.y as usize].organism.is_none() {
                sim.simulation_board.cells[organism.position.x as usize - 1][organism.position.y as usize].organism = Some(organism.clone());
                sim.simulation_board.cells[organism.position.x as usize - 1][organism.position.y as usize].organism.as_mut().unwrap().position.x -= 1;
                sim.simulation_board.cells[organism.position.x as usize][organism.position.y as usize].organism = None;
            }
        }),
    }
];

const SIMULATION_CONFIG: SimulationConfig = SimulationConfig {
    brain_config: BrainConfig {
        unique_input_neurons_count: 3,
        unique_hidden_neurons_count: 3,
        unique_output_neurons_count: 4,
        synapses: 5,
        neuron_types: NEURON_TYPES_ARRAY,
    },
    board_config: BoardConfig {
        width: 50,
        height: 50,
    },
    evolution_config: EvolutionConfig {
        reproduction_factor: 1,
        mutation_probability: 0.01,
    },
    initial_population: 300,
    steps_per_second: 30,
    max_generations: 0, // 0 = unlimited
    steps_per_generation: 300,
    selection_method: |sim, organism| -> bool {
        organism.position.x < (sim.simulation_config.board_config.width / 2) as i32
    },
};

fn main() {
    let mut simulation = SimulationState::new(&SIMULATION_CONFIG);
    simulation.run();
}
