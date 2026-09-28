/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.parquet.benchmarks;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.CodecPool;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionOutputStream;
import org.apache.hadoop.io.compress.Compressor;
import org.apache.hadoop.io.compress.Decompressor;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Compares three ways of decompressing a Parquet data page, to quantify the per-page allocation of
 * each. Run with the {@code gc} profiler and read {@code gc.alloc.rate.norm} (bytes per op):
 *
 * <pre>
 *   run.sh PageDecompressorBenchmark -prof gc -p codecName=SNAPPY
 * </pre>
 *
 * <ul>
 *   <li>{@link #streamPerPage} -- the current heap path: a {@code DecompressorStream} (and its
 *       internal buffer) is allocated per page.
 *   <li>{@link #reusedBuffer} -- one-shot decompression into a buffer reused across pages.
 *   <li>{@link #freshBufferPerPage} -- one-shot decompression into a new buffer per page.
 * </ul>
 *
 * All three are fed a {@link BytesInput} (as the real decompressor is), so the one-shot paths
 * include the {@code toByteArray()} input copy they incur, for a fair comparison.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(1)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public class PageDecompressorBenchmark {

  @Param({"SNAPPY", "GZIP", "ZSTD", "LZ4_RAW"})
  public CompressionCodecName codecName;

  /** Uncompressed page size in bytes (default parquet page size is ~1 MB; 128 KB is a common one). */
  @Param({"131072"})
  public int pageSize;

  private CompressionCodec codec;
  private Decompressor decompressor;
  private byte[] compressed;
  private int decompressedSize;

  // Reused across pages by the reusedBuffer strategy, and as the read sink of the stream strategy.
  private byte[] reusedBuffer;
  private byte[] readSink;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    Configuration conf = new Configuration();
    Class<?> codecClass = Class.forName(codecName.getHadoopCompressionCodecClassName());
    codec = (CompressionCodec) ReflectionUtils.newInstance(codecClass, conf);

    // A low-entropy page so it compresses like real (dictionary-ish) column data.
    byte[] raw = new byte[pageSize];
    Random random = new Random(42);
    for (int i = 0; i < raw.length; i++) {
      raw[i] = (byte) random.nextInt(16);
    }
    decompressedSize = raw.length;

    Compressor compressor = CodecPool.getCompressor(codec);
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (CompressionOutputStream cos = codec.createOutputStream(out, compressor)) {
        cos.write(raw);
        cos.finish();
      }
      compressed = out.toByteArray();
    } finally {
      CodecPool.returnCompressor(compressor);
    }

    decompressor = CodecPool.getDecompressor(codec);
    reusedBuffer = new byte[decompressedSize];
    readSink = new byte[8192];
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    if (decompressor != null) {
      CodecPool.returnDecompressor(decompressor);
    }
  }

  private BytesInput compressedPage() {
    // A fresh BytesInput per page, as ColumnChunkPageReadStore hands to the decompressor.
    return BytesInput.from(ByteBuffer.wrap(compressed));
  }

  @Benchmark
  public void streamPerPage(Blackhole bh) throws IOException {
    decompressor.reset();
    InputStream is = codec.createInputStream(compressedPage().toInputStream(), decompressor);
    try {
      int n;
      while ((n = is.read(readSink)) != -1) {
        bh.consume(readSink);
        bh.consume(n);
      }
    } finally {
      is.close();
    }
  }

  @Benchmark
  public void reusedBuffer(Blackhole bh) throws IOException {
    decompressor.reset();
    byte[] input = compressedPage().toByteArray();
    decompressor.setInput(input, 0, input.length);
    int offset = 0;
    while (offset < decompressedSize) {
      int count = decompressor.decompress(reusedBuffer, offset, decompressedSize - offset);
      if (count == 0) {
        break;
      }
      offset += count;
    }
    bh.consume(reusedBuffer);
  }

  @Benchmark
  public void freshBufferPerPage(Blackhole bh) throws IOException {
    decompressor.reset();
    byte[] input = compressedPage().toByteArray();
    decompressor.setInput(input, 0, input.length);
    byte[] output = new byte[decompressedSize];
    int offset = 0;
    while (offset < decompressedSize) {
      int count = decompressor.decompress(output, offset, decompressedSize - offset);
      if (count == 0) {
        break;
      }
      offset += count;
    }
    bh.consume(output);
  }
}
