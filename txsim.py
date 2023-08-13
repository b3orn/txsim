import collections
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg.linalg as la
import numpy.fft as fft


GRAY_CODES = {
    2: [0b0, 0b1],
    4: [0b00, 0b01, 0b11, 0b10],
    8: [0b000, 0b001, 0b011, 0b111, 0b101, 0b100, 0b110, 0b010],
    16: [0b0000, 0b0001, 0b0011, 0b0111,
         0b0101, 0b1101, 0b1111, 0b1011,
         0b1001, 0b1000, 0b1010, 0b1110,
         0b1100, 0b0100, 0b0110, 0b0010],
}
BARKER_11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
BARKER_5 = np.array([1, 1, 1, -1, 1])


def rect(t, t_s):
    return np.where(np.abs(t) < t_s/2, 1, 0)


def sqrt_raised_cosine(t, a, t_s):
    x = 4*a*t/t_s * np.cos(np.pi*(1 + a)*t/t_s) + np.sin(np.pi*(1-a)*t/t_s)
    y = np.pi*t/t_s * (1 - (4*a*t/t_s)**2)
    return np.divide(x, y, where=y != 0, out=np.zeros_like(t))


def spreading_code(t, s, t_s, f_s):
    if s == 5:
        code = BARKER_5
    elif s == 11:
        code = BARKER_11
    else:
        return 1

    t_c = t_s/s
    spreading_code = np.zeros_like(t)
    pulse = rect(t - 0.5 * t_c, t_c)
    n = int(t_c * f_s)
    for i, c in enumerate(code):
        spreading_code += c * np.roll(pulse, i * n)

    return spreading_code


def convolve(a, b, *args, **kwargs):
    # for some reason numpy's convolve is weird and the second paramter has to be flipped for an actual convolution
    return np.convolve(a, np.flip(b), *args, **kwargs)


def average_energy(m, mapping):
    return 1/m * np.sum(np.abs(np.array(list(mapping.values()))) ** 2)


def norm_mapping(m, mapping):
    k = np.sqrt(1 / average_energy(m, mapping))

    return {code: k * symbol for code, symbol in mapping.items()}


def on_off(m):
    assert m == 2
    return {
        0: 0,
        1: np.sqrt(2)
    }


def m_ask(m):
    assert 2 ** int(np.log2(m)) == m

    codes = GRAY_CODES[m]
    mapping = {codes[i]: m / 2 - i - 0.5 for i in range(m)}

    return norm_mapping(m, mapping)


def m_psk(m, rotate=True):
    assert 2 ** int(np.log2(m)) == m

    phase = 2 * np.pi / m
    phase_offset = np.pi / m if rotate and m > 2 else 0
    codes = GRAY_CODES[m]
    mapping = {codes[i]: np.exp(1j * (i * phase + phase_offset)) for i in range(m)}

    return norm_mapping(m, mapping)


def m_qam(m):
    if m == 4:
        return m_psk(m)
    elif m == 8:
        codes = GRAY_CODES[m]
        mapping = {
            codes[0]: 1,
            codes[1]: 1 + 1j,
            codes[2]: 1j,
            codes[3]: -1 + 1j,
            codes[4]: -1,
            codes[5]: -1 - 1j,
            codes[6]: -1j,
            codes[7]: 1 - 1j,
        }
    elif m == 16:
        mapping = {
            0b0101: 1/2 + 1j/2,
            0b0111: -1/2 + 1j/2,
            0b1101: 1/2 - 1j/2,
            0b1111: -1/2 - 1j/2,
            0b0001: 1/2 + 3j/2,
            0b0011: -1/2 + 3j/2,
            0b0110: -3/2 + 1j/2,
            0b1110: -3/2 - 1j/2,
            0b1011: -1/2 - 3j/2,
            0b1001: 1/2 - 3j/2,
            0b0100: 3/2 + 1j/2,
            0b1100: 3/2 - 1j/2,
            0b0000: 3/2 + 3j/2,
            0b0010: -3/2 + 3j/2,
            0b1010: -3/2 - 3j/2,
            0b1000: 3/2 - 3j/2,
        }
    else:
        raise NotImplementedError
    
    return norm_mapping(m, mapping)


def fir_filter(signal, h, n):
    result = signal.data * h[0]

    for i, hn in enumerate(h[1:]):
        result[(i+1)*n:] += (signal.data * hn)[:-(i+1)*n]

    return Signal(signal.t, result)


def lowpass_filter(signal, cutoff, f_s):
    Y = fft.fft(signal.data)
    freq = f_s * np.fft.fftfreq(signal.t.shape[0])
    Y[np.abs(freq) > cutoff] = 0
    return Signal(signal.t, fft.ifft(Y))


class Simulation:
    def __init__(self, f_s):
        self.f_s = f_s

    def t(self, duration):
        return np.linspace(0, duration, int(np.ceil(duration * self.f_s)))

    def create_channel(self, t_s, snr=0, h=None, noisy=True):
        return Channel(
            t_s=t_s,
            snr=snr,
            h=h,
            noisy=noisy,
            simulation=self,
        )

    def create_transmitter(self, f_c, t_s, mapping, spreading_factor=1, delay_quadrature=False, pulse=rect):
        return Transmitter(
            f_c=f_c,
            t_s=t_s,
            mapping=mapping,
            spreading_factor=spreading_factor,
            delay_quadrature=delay_quadrature,
            pulse=pulse,
            simulation=self,
        )

    def create_receiver(self, f_c, t_s, mapping, equalizer=None, spreading_factor=1, delay_quadrature=False, phase_offset=0, pulse=rect):
        return Receiver(
            f_c=f_c,
            t_s=t_s,
            mapping=mapping,
            equalizer=equalizer,
            spreading_factor=spreading_factor,
            delay_quadrature=delay_quadrature,
            phase_offset=phase_offset,
            pulse=pulse,
            simulation=self,
        )

    def create_mapping(self, size, mapping):
        types = {
            "onoff": lambda: Mapping(size, on_off),
            "ask": lambda: Mapping(size, m_ask),
            "psk": lambda: Mapping(size, m_psk),
            "apsk": lambda: Mapping(size, m_psk),
            "dpsk": lambda: DifferentialPSKMapping(size, m_psk),
            "dapsk": lambda: DifferentialPSKMapping(size, m_psk),
            "qam": lambda: Mapping(size, m_qam),
            "fsk": lambda: FSKMapping,
        }
        return types[mapping]()

    def create_equalizer(self, type, channel, n=1):
        types = {
            "zf": ZFEqualizer,
            "mmse": MMSEEqualizer,
            "dfe": DecisionFeedbackEqualizer,
        }
        return types[type](
            channel=channel,
            n=n,
            simulation=self
        )

    def create_scope(self, **kwargs):
        return Scope(self, **kwargs)


class Observable:
    def __init__(self):
        self.scopes = collections.defaultdict(list)

    def attach(self, scope, signal):
        self.scopes[signal].append(scope)

    def detach(self, scope, signal):
        self.scopes[signal].remove(scope)

    def trigger(self, signal, data):
        for scope in self.scopes[signal]:
            scope.trigger(self, signal, data)


class Mapping(Observable):
    def __init__(self, size, mapping):
        super().__init__()
        self.mapping = mapping(size)
        self.size = size
        self.bitsize = int(np.log2(self.size))

        assert 2 ** self.bitsize == self.size

        self.mask = 1
        for i in range(self.bitsize - 1):
            self.mask <<= 1
            self.mask |= 1

    @property
    def constellation(self):
        return np.array(list(self.mapping.values()))

    def deconstruct(self, data):
        codes = []
        for element in data:
            n = 8
            while n > 0:
                x = element & self.mask
                codes.append(x)
                element >>= self.bitsize
                n -= self.bitsize

        return codes

    def reconstruct(self, codes):
        result = []
        element = 0
        n = 0
        for code in codes:
            element |= code << n
            n += self.bitsize
            if n >= 8:
                result.append(element)
                element = 0
                n = 0

        return bytes(result)

    def map_symbols(self, codes):
        return np.array([self.mapping[code] for code in codes])

    def demap_symbols(self, received_symbols):
        codes = []
        for received_symbol in received_symbols:
            min_distance = None
            detected_symbol = None
            for code, symbol in self.mapping.items():
                distance = np.abs(received_symbol - symbol)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    detected_symbol = code
            codes.append(detected_symbol)

        return codes

    def differential_encode(self, symbols):
        return symbols

    def differential_decode(self, symbols):
        return symbols

    def encode(self, data):
        deconstructed = self.deconstruct(data)
        symbols = self.map_symbols(deconstructed)
        self.trigger("symbols", symbols)
        encoded = self.differential_encode(symbols)
        self.trigger("encoded_symbols", encoded)
        return encoded

    def decode(self, symbols):
        self.trigger("detected_symbols", symbols)
        decoded = self.differential_decode(symbols)
        self.trigger("decoded_symbols", decoded)
        codes = self.demap_symbols(decoded)
        return self.reconstruct(codes)


class DifferentialPSKMapping(Mapping):
    def differential_encode(self, symbols):
        return np.insert(symbols, 0, 1).cumprod()

    def differential_decode(self, symbols):
        return symbols[1:] * symbols[:-1].conjugate()


class FSKMapping(Mapping):
    pass


class Equalizer:
    def __init__(self, channel, n=1, simulation=None):
        self.channel = channel
        self.n = n
        self.simulation = simulation

    def reset(self):
        pass

    def equalize(self, data):
        raise NotImplementedError


class LinearEqualizer(Equalizer):
    def __init__(self, channel, n=1, simulation=None):
        super().__init__(channel, n, simulation)
        self.reset()

    def reset(self):
        if self.n == 1:
            self.e = np.array([1])
        else:
            self.e = self.parameters()

    def equalize(self, signal):
        return fir_filter(signal, self.e, int(self.channel.t_s * self.channel.simulation.f_s))

    def convolution_matrix(self):
        d = len(self.channel.h) - 1
        p = self.n - 1
        H = np.zeros((p + 1, d + p + 1))

        for i in range(p + 1):
            for j in range(d + 1):
                H[i, i+j] = self.channel.h[j]

        in0 = np.eye(d + p + 1, 1)

        return H.T, in0

    def parameters(self):
        raise NotImplementedError


class ZFEqualizer(LinearEqualizer):
    def parameters(self):
        H, in0 = self.convolution_matrix()
        return la.inv(H.T @ H) @ H.T @ in0


class MMSEEqualizer(LinearEqualizer):
    def parameters(self):
        H, in0 = self.convolution_matrix()
        return la.inv(H.T @ H + 10 ** (-self.channel.snr/10) * np.eye(self.n)) @ H.T @ in0


class DecisionFeedbackEqualizer(Equalizer):
    pass


class Signal:
    def __init__(self, t, data):
        self.t = t
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def real(self):
        return Signal(self.t, self.data.real)

    @property
    def imag(self):
        return Signal(self.t, self.data.imag)

    def __add__(self, other):
        return Signal(self.t, self.data + other)

    def __mul__(self, other):
        return Signal(self.t, self.data * other)


class Channel:
    def __init__(self, t_s, snr=0, h=None, noisy=True, simulation=None):
        self.t_s = t_s
        self.snr = snr
        self.h = h if h is not None else np.array([1])
        self.noisy = noisy
        self.simulation = simulation

    def delay_rms(self):
        tau = self.t_s * np.linspace(0, len(self.h) - 1, len(self.h))
        tau_mean = np.sum(tau * self.h**2)/np.sum(self.h**2)

        return np.sqrt(np.sum((tau - tau_mean)**2 * self.h**2)/np.sum(self.h**2))

    def frequency_selective(self, t_s):
        return self.delay_rms() > 0.1 * t_s

    def transmit(self, signal):
        if self.noisy:
            noise = np.sqrt(2) * np.random.normal(0, 10 ** (-self.snr/20), signal.data.shape)
        else:
            noise = 0

        return fir_filter(signal, self.h, n = int(self.t_s * self.simulation.f_s)) + noise


class Transmitter(Observable):
    def __init__(self, f_c, t_s, mapping, spreading_factor=1, delay_quadrature=False, pulse=rect, simulation=None):
        super().__init__()
        self.f_c = f_c
        self.t_s = t_s
        self.mapping = mapping
        self.spreading_factor = spreading_factor
        self.delay_quadrature = delay_quadrature
        self.pulse = pulse or rect
        self.simulation = simulation

    def pulse_filter(self, symbols):
        t = self.simulation.t(self.t_s * (len(symbols) + (1 if self.delay_quadrature else 0)))
        s_bb = np.zeros_like(t, dtype=np.complex128)
        pulse = self.pulse(t - 0.5 * self.t_s, self.t_s) * spreading_code(t, self.spreading_factor, self.t_s, self.simulation.f_s)
        n = int(self.t_s * self.simulation.f_s)

        for i, symbol in enumerate(symbols):
            s_bb += symbol * np.roll(pulse, i * n)

        if self.delay_quadrature:
            s_bb = s_bb.real + 1j * np.roll(s_bb.imag, int(np.ceil(0.5 * n)))

        signal = Signal(t, s_bb)

        self.trigger("baseband", signal)

        return signal

    def modulate(self, signal):
        modulated = (signal * np.sqrt(2) * np.exp(1j * 2 * np.pi * self.f_c * signal.t)).real

        self.trigger("modulated", modulated)

        return modulated

    def transmit_baseband(self, data):
        symbols = self.mapping.encode(data)
        return self.pulse_filter(symbols)

    def transmit_data(self, data):
        baseband = self.transmit_baseband(data)
        return self.modulate(baseband)

    def transmit_pilot(self, n):
        t = self.simulation.t(n * self.t_s)
        pilot = Signal(t, spreading_code(t, 11, self.t_s, self.simulation.f_s))

        self.trigger("baseband", pilot)

        return self.modulate(pilot)


class Receiver(Observable):
    def __init__(self, f_c, t_s, mapping, equalizer=None, spreading_factor=1, delay_quadrature=False, phase_offset=0, pulse=rect, simulation=None):
        super().__init__()
        self.f_c = f_c
        self.t_s = t_s
        self.mapping = mapping
        self.equalizer = equalizer
        self.spreading_factor = spreading_factor
        self.delay_quadrature = delay_quadrature
        self.phase_offset = phase_offset
        self.pulse = pulse or rect
        self.simulation = simulation

    def demodulate(self, signal):
        demoduldated = signal * np.sqrt(2) * np.exp(-1j * 2 * np.pi * self.f_c * signal.t - 1j * self.phase_offset)

        self.trigger("demodulated", demoduldated)

        return demoduldated

    def filter(self, signal):
        cutoff = 2 * self.spreading_factor / self.t_s
        filtered = lowpass_filter(signal, cutoff, self.simulation.f_s)

        self.trigger("filtered", filtered)

        return filtered

    def equalize(self, data):
        if not self.equalizer:
            return data

        equalized = self.equalizer.equalize(data)

        self.trigger("equalized", equalized)

        return equalized

    def pulse_filter(self, signal):
        n = int(self.t_s * self.simulation.f_s)
        n_symbols = int(len(signal) / n) - (1 if self.delay_quadrature else 0)
        pulse = self.pulse(signal.t - 0.5 * self.t_s, self.t_s) * spreading_code(signal.t, self.spreading_factor, self.t_s, self.simulation.f_s)

        data = signal.data
        if self.delay_quadrature:
            data = data.real + 1j * np.roll(data.imag, int(np.ceil(-0.5 * n)))

        result = np.zeros_like(signal.t, dtype=np.complex128)
        for i in range(n_symbols):
            result += data * np.roll(pulse, i * n)

        filtered = Signal(signal.t, result)

        self.trigger("pulse_filtered", filtered)

        return filtered

    def detect_symbols(self, signal):
        n = int(self.t_s * self.simulation.f_s)
        n_symbols = int(len(signal) / n) - (1 if self.delay_quadrature else 0)
        symbols = np.zeros(n_symbols, dtype=np.complex128)
        pulse = self.pulse(signal.t - 0.5 * self.t_s, self.t_s)

        for i in range(n_symbols):
            symbols[i] = 1/n * np.sum(signal.data * np.roll(pulse, i * n))

        self.trigger("detected_symbols", symbols)

        return symbols

    def receive_baseband(self, signal):
        self.trigger("received", signal)
        demodulated = self.demodulate(signal)
        filtered = self.filter(demodulated)
        return self.equalize(filtered)

    def receive_from_baseband(self, signal):
        filtered = self.pulse_filter(signal)
        detected_symbols = self.detect_symbols(filtered)
        return self.mapping.decode(detected_symbols)

    def receive_data(self, signal):
        baseband_signal = self.receive_baseband(signal)
        return self.receive_from_baseband(baseband_signal)

    def estimate_channel(self, signal, channel):
        self.trigger("received", signal)

        demodulated = self.demodulate(signal)
        baseband = self.filter(demodulated)
        t = self.simulation.t(self.t_s)
        n = int(channel.t_s * self.simulation.f_s)
        pulse = spreading_code(t, 11, self.t_s, self.simulation.f_s)
        q = convolve(baseband.data, pulse, mode="valid") / len(t)
        h = q.real[::n]

        self.trigger("channel_response", Signal(signal.t[:len(q)], q))

        return h


class Scope:
    def __init__(self, simulation=None, **kwargs):
        self.simulation = simulation
        self.kwargs = kwargs
        self.handlers = collections.defaultdict(list)
        self.plotter = {
            "xy": self.plot_constellation,
            "t": self.plot_time,
            "f": self.plot_frequency,
        }

    def attach(self, object, signal, domain="t", **kwargs):
        if not len(self.handlers[(object, signal)]):
            object.attach(self, signal)
        self.handlers[(object, signal)].append((domain, kwargs))

    def detach(self, object, signal, domain="t"):
        handlers = self.handlers[(object, signal)]
        self.handlers[(object, signal)] = [h for h in handlers if h[0] != domain]

        if not len(self.handlers[(object, signal)]):
            object.detach(self, signal)

    def trigger(self, object, signal, data):
        for domain, kwargs in self.handlers[(object, signal)]:
            self.plotter[domain](data, **kwargs)

    def plot_constellation(self, data, title="", reference=None, **kwargs):
        if isinstance(data, Signal):
            data = data.data

        plt.figure(figsize=kwargs.get("figsize", self.kwargs.get("figsize")))
        plt.title(title)
        if reference is not None:
            plt.plot(reference.real, reference.imag, label="Reference", marker="o", color="tab:orange", linestyle="", )

        plt.plot(data.real, data.imag, label="Signal", marker="x", color="tab:blue", linestyle=kwargs.get("linestyle", ""))
        plt.legend()
        plt.grid()
        plt.show()

    def plot_time(self, data, title="", imag=True, **kwargs):
        plt.figure(figsize=kwargs.get("figsize", self.kwargs.get("figsize")))
        if imag:
            plt.subplot(211)

        plt.title(title)
        plt.plot(data.t, data.data.real, label="I", color="tab:blue")
        plt.legend()
        plt.grid()
        plt.xlabel("t [s]")

        if imag:
            plt.subplot(212)
            plt.plot(data.t, data.data.imag, label="Q", color="tab:orange")
            plt.legend()
            plt.grid()
            plt.xlabel("t [s]")

        plt.show()

    def plot_frequency(self, data, title="", **kwargs):
        plt.figure(figsize=kwargs.get("figsize", self.kwargs.get("figsize")))
        plt.title(title)
        freqs = self.simulation.f_s * fft.fftfreq(len(data))
        Y = 1/len(data) * np.abs(fft.fft(data.data))
        if kwargs.get("db") or kwargs.get("dB"):
            Y = 10 * np.log10(Y)
            plt.plot(freqs, Y, color="tab:blue")
            plt.ylabel("dB")
        else:
            plt.plot(freqs, Y, color="tab:blue")
        plt.xlabel("f [Hz]")
        plt.xlim(kwargs.get("xlim"))
        plt.ylim(ymin=kwargs.get("ymin"))
        plt.grid()
        plt.show()
