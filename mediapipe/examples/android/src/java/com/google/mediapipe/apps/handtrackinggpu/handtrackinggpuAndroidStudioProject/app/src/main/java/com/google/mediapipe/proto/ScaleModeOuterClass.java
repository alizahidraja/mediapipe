// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/gpu/scale_mode.proto

package mediapipe;

public final class ScaleModeOuterClass {
  private ScaleModeOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  public interface ScaleModeOrBuilder extends
      // @@protoc_insertion_point(interface_extends:mediapipe.ScaleMode)
      com.google.protobuf.MessageOrBuilder {
  }
  /**
   * <pre>
   * We wrap the enum in a message to avoid namespace collisions.
   * </pre>
   *
   * Protobuf type {@code mediapipe.ScaleMode}
   */
  public  static final class ScaleMode extends
      com.google.protobuf.GeneratedMessageV3 implements
      // @@protoc_insertion_point(message_implements:mediapipe.ScaleMode)
      ScaleModeOrBuilder {
    // Use ScaleMode.newBuilder() to construct.
    private ScaleMode(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
      super(builder);
    }
    private ScaleMode() {
    }

    @java.lang.Override
    public final com.google.protobuf.UnknownFieldSet
    getUnknownFields() {
      return this.unknownFields;
    }
    private ScaleMode(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      this();
      com.google.protobuf.UnknownFieldSet.Builder unknownFields =
          com.google.protobuf.UnknownFieldSet.newBuilder();
      try {
        boolean done = false;
        while (!done) {
          int tag = input.readTag();
          switch (tag) {
            case 0:
              done = true;
              break;
            default: {
              if (!parseUnknownField(input, unknownFields,
                                     extensionRegistry, tag)) {
                done = true;
              }
              break;
            }
          }
        }
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw e.setUnfinishedMessage(this);
      } catch (java.io.IOException e) {
        throw new com.google.protobuf.InvalidProtocolBufferException(
            e).setUnfinishedMessage(this);
      } finally {
        this.unknownFields = unknownFields.build();
        makeExtensionsImmutable();
      }
    }
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return mediapipe.ScaleModeOuterClass.internal_static_mediapipe_ScaleMode_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return mediapipe.ScaleModeOuterClass.internal_static_mediapipe_ScaleMode_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              mediapipe.ScaleModeOuterClass.ScaleMode.class, mediapipe.ScaleModeOuterClass.ScaleMode.Builder.class);
    }

    /**
     * <pre>
     * This enum mirrors the ScaleModes supported by Quad Renderer.
     * </pre>
     *
     * Protobuf enum {@code mediapipe.ScaleMode.Mode}
     */
    public enum Mode
        implements com.google.protobuf.ProtocolMessageEnum {
      /**
       * <code>DEFAULT = 0;</code>
       */
      DEFAULT(0),
      /**
       * <pre>
       * Stretch the frame to the exact provided output dimensions.
       * </pre>
       *
       * <code>STRETCH = 1;</code>
       */
      STRETCH(1),
      /**
       * <pre>
       * Scale the frame up to fit the drawing area, preserving aspect ratio; may
       * letterbox.
       * </pre>
       *
       * <code>FIT = 2;</code>
       */
      FIT(2),
      /**
       * <pre>
       * Scale the frame up to fill the drawing area, preserving aspect ratio; may
       * crop.
       * </pre>
       *
       * <code>FILL_AND_CROP = 3;</code>
       */
      FILL_AND_CROP(3),
      ;

      /**
       * <code>DEFAULT = 0;</code>
       */
      public static final int DEFAULT_VALUE = 0;
      /**
       * <pre>
       * Stretch the frame to the exact provided output dimensions.
       * </pre>
       *
       * <code>STRETCH = 1;</code>
       */
      public static final int STRETCH_VALUE = 1;
      /**
       * <pre>
       * Scale the frame up to fit the drawing area, preserving aspect ratio; may
       * letterbox.
       * </pre>
       *
       * <code>FIT = 2;</code>
       */
      public static final int FIT_VALUE = 2;
      /**
       * <pre>
       * Scale the frame up to fill the drawing area, preserving aspect ratio; may
       * crop.
       * </pre>
       *
       * <code>FILL_AND_CROP = 3;</code>
       */
      public static final int FILL_AND_CROP_VALUE = 3;


      public final int getNumber() {
        return value;
      }

      /**
       * @deprecated Use {@link #forNumber(int)} instead.
       */
      @java.lang.Deprecated
      public static Mode valueOf(int value) {
        return forNumber(value);
      }

      public static Mode forNumber(int value) {
        switch (value) {
          case 0: return DEFAULT;
          case 1: return STRETCH;
          case 2: return FIT;
          case 3: return FILL_AND_CROP;
          default: return null;
        }
      }

      public static com.google.protobuf.Internal.EnumLiteMap<Mode>
          internalGetValueMap() {
        return internalValueMap;
      }
      private static final com.google.protobuf.Internal.EnumLiteMap<
          Mode> internalValueMap =
            new com.google.protobuf.Internal.EnumLiteMap<Mode>() {
              public Mode findValueByNumber(int number) {
                return Mode.forNumber(number);
              }
            };

      public final com.google.protobuf.Descriptors.EnumValueDescriptor
          getValueDescriptor() {
        return getDescriptor().getValues().get(ordinal());
      }
      public final com.google.protobuf.Descriptors.EnumDescriptor
          getDescriptorForType() {
        return getDescriptor();
      }
      public static final com.google.protobuf.Descriptors.EnumDescriptor
          getDescriptor() {
        return mediapipe.ScaleModeOuterClass.ScaleMode.getDescriptor().getEnumTypes().get(0);
      }

      private static final Mode[] VALUES = values();

      public static Mode valueOf(
          com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
        if (desc.getType() != getDescriptor()) {
          throw new java.lang.IllegalArgumentException(
            "EnumValueDescriptor is not for this type.");
        }
        return VALUES[desc.getIndex()];
      }

      private final int value;

      private Mode(int value) {
        this.value = value;
      }

      // @@protoc_insertion_point(enum_scope:mediapipe.ScaleMode.Mode)
    }

    private byte memoizedIsInitialized = -1;
    public final boolean isInitialized() {
      byte isInitialized = memoizedIsInitialized;
      if (isInitialized == 1) return true;
      if (isInitialized == 0) return false;

      memoizedIsInitialized = 1;
      return true;
    }

    public void writeTo(com.google.protobuf.CodedOutputStream output)
                        throws java.io.IOException {
      unknownFields.writeTo(output);
    }

    public int getSerializedSize() {
      int size = memoizedSize;
      if (size != -1) return size;

      size = 0;
      size += unknownFields.getSerializedSize();
      memoizedSize = size;
      return size;
    }

    private static final long serialVersionUID = 0L;
    @java.lang.Override
    public boolean equals(final java.lang.Object obj) {
      if (obj == this) {
       return true;
      }
      if (!(obj instanceof mediapipe.ScaleModeOuterClass.ScaleMode)) {
        return super.equals(obj);
      }
      mediapipe.ScaleModeOuterClass.ScaleMode other = (mediapipe.ScaleModeOuterClass.ScaleMode) obj;

      boolean result = true;
      result = result && unknownFields.equals(other.unknownFields);
      return result;
    }

    @java.lang.Override
    public int hashCode() {
      if (memoizedHashCode != 0) {
        return memoizedHashCode;
      }
      int hash = 41;
      hash = (19 * hash) + getDescriptorForType().hashCode();
      hash = (29 * hash) + unknownFields.hashCode();
      memoizedHashCode = hash;
      return hash;
    }

    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(
        com.google.protobuf.ByteString data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(
        com.google.protobuf.ByteString data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(byte[] data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(
        byte[] data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseDelimitedFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseDelimitedFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(
        com.google.protobuf.CodedInputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static mediapipe.ScaleModeOuterClass.ScaleMode parseFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }

    public Builder newBuilderForType() { return newBuilder(); }
    public static Builder newBuilder() {
      return DEFAULT_INSTANCE.toBuilder();
    }
    public static Builder newBuilder(mediapipe.ScaleModeOuterClass.ScaleMode prototype) {
      return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
    }
    public Builder toBuilder() {
      return this == DEFAULT_INSTANCE
          ? new Builder() : new Builder().mergeFrom(this);
    }

    @java.lang.Override
    protected Builder newBuilderForType(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      Builder builder = new Builder(parent);
      return builder;
    }
    /**
     * <pre>
     * We wrap the enum in a message to avoid namespace collisions.
     * </pre>
     *
     * Protobuf type {@code mediapipe.ScaleMode}
     */
    public static final class Builder extends
        com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
        // @@protoc_insertion_point(builder_implements:mediapipe.ScaleMode)
        mediapipe.ScaleModeOuterClass.ScaleModeOrBuilder {
      public static final com.google.protobuf.Descriptors.Descriptor
          getDescriptor() {
        return mediapipe.ScaleModeOuterClass.internal_static_mediapipe_ScaleMode_descriptor;
      }

      protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
          internalGetFieldAccessorTable() {
        return mediapipe.ScaleModeOuterClass.internal_static_mediapipe_ScaleMode_fieldAccessorTable
            .ensureFieldAccessorsInitialized(
                mediapipe.ScaleModeOuterClass.ScaleMode.class, mediapipe.ScaleModeOuterClass.ScaleMode.Builder.class);
      }

      // Construct using mediapipe.ScaleModeOuterClass.ScaleMode.newBuilder()
      private Builder() {
        maybeForceBuilderInitialization();
      }

      private Builder(
          com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
        super(parent);
        maybeForceBuilderInitialization();
      }
      private void maybeForceBuilderInitialization() {
        if (com.google.protobuf.GeneratedMessageV3
                .alwaysUseFieldBuilders) {
        }
      }
      public Builder clear() {
        super.clear();
        return this;
      }

      public com.google.protobuf.Descriptors.Descriptor
          getDescriptorForType() {
        return mediapipe.ScaleModeOuterClass.internal_static_mediapipe_ScaleMode_descriptor;
      }

      public mediapipe.ScaleModeOuterClass.ScaleMode getDefaultInstanceForType() {
        return mediapipe.ScaleModeOuterClass.ScaleMode.getDefaultInstance();
      }

      public mediapipe.ScaleModeOuterClass.ScaleMode build() {
        mediapipe.ScaleModeOuterClass.ScaleMode result = buildPartial();
        if (!result.isInitialized()) {
          throw newUninitializedMessageException(result);
        }
        return result;
      }

      public mediapipe.ScaleModeOuterClass.ScaleMode buildPartial() {
        mediapipe.ScaleModeOuterClass.ScaleMode result = new mediapipe.ScaleModeOuterClass.ScaleMode(this);
        onBuilt();
        return result;
      }

      public Builder clone() {
        return (Builder) super.clone();
      }
      public Builder setField(
          com.google.protobuf.Descriptors.FieldDescriptor field,
          Object value) {
        return (Builder) super.setField(field, value);
      }
      public Builder clearField(
          com.google.protobuf.Descriptors.FieldDescriptor field) {
        return (Builder) super.clearField(field);
      }
      public Builder clearOneof(
          com.google.protobuf.Descriptors.OneofDescriptor oneof) {
        return (Builder) super.clearOneof(oneof);
      }
      public Builder setRepeatedField(
          com.google.protobuf.Descriptors.FieldDescriptor field,
          int index, Object value) {
        return (Builder) super.setRepeatedField(field, index, value);
      }
      public Builder addRepeatedField(
          com.google.protobuf.Descriptors.FieldDescriptor field,
          Object value) {
        return (Builder) super.addRepeatedField(field, value);
      }
      public Builder mergeFrom(com.google.protobuf.Message other) {
        if (other instanceof mediapipe.ScaleModeOuterClass.ScaleMode) {
          return mergeFrom((mediapipe.ScaleModeOuterClass.ScaleMode)other);
        } else {
          super.mergeFrom(other);
          return this;
        }
      }

      public Builder mergeFrom(mediapipe.ScaleModeOuterClass.ScaleMode other) {
        if (other == mediapipe.ScaleModeOuterClass.ScaleMode.getDefaultInstance()) return this;
        this.mergeUnknownFields(other.unknownFields);
        onChanged();
        return this;
      }

      public final boolean isInitialized() {
        return true;
      }

      public Builder mergeFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws java.io.IOException {
        mediapipe.ScaleModeOuterClass.ScaleMode parsedMessage = null;
        try {
          parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
          parsedMessage = (mediapipe.ScaleModeOuterClass.ScaleMode) e.getUnfinishedMessage();
          throw e.unwrapIOException();
        } finally {
          if (parsedMessage != null) {
            mergeFrom(parsedMessage);
          }
        }
        return this;
      }
      public final Builder setUnknownFields(
          final com.google.protobuf.UnknownFieldSet unknownFields) {
        return super.setUnknownFields(unknownFields);
      }

      public final Builder mergeUnknownFields(
          final com.google.protobuf.UnknownFieldSet unknownFields) {
        return super.mergeUnknownFields(unknownFields);
      }


      // @@protoc_insertion_point(builder_scope:mediapipe.ScaleMode)
    }

    // @@protoc_insertion_point(class_scope:mediapipe.ScaleMode)
    private static final mediapipe.ScaleModeOuterClass.ScaleMode DEFAULT_INSTANCE;
    static {
      DEFAULT_INSTANCE = new mediapipe.ScaleModeOuterClass.ScaleMode();
    }

    public static mediapipe.ScaleModeOuterClass.ScaleMode getDefaultInstance() {
      return DEFAULT_INSTANCE;
    }

    @java.lang.Deprecated public static final com.google.protobuf.Parser<ScaleMode>
        PARSER = new com.google.protobuf.AbstractParser<ScaleMode>() {
      public ScaleMode parsePartialFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws com.google.protobuf.InvalidProtocolBufferException {
          return new ScaleMode(input, extensionRegistry);
      }
    };

    public static com.google.protobuf.Parser<ScaleMode> parser() {
      return PARSER;
    }

    @java.lang.Override
    public com.google.protobuf.Parser<ScaleMode> getParserForType() {
      return PARSER;
    }

    public mediapipe.ScaleModeOuterClass.ScaleMode getDefaultInstanceForType() {
      return DEFAULT_INSTANCE;
    }

  }

  private static final com.google.protobuf.Descriptors.Descriptor
    internal_static_mediapipe_ScaleMode_descriptor;
  private static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_mediapipe_ScaleMode_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\036mediapipe/gpu/scale_mode.proto\022\tmediap" +
      "ipe\"I\n\tScaleMode\"<\n\004Mode\022\013\n\007DEFAULT\020\000\022\013\n" +
      "\007STRETCH\020\001\022\007\n\003FIT\020\002\022\021\n\rFILL_AND_CROP\020\003"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
        }, assigner);
    internal_static_mediapipe_ScaleMode_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_mediapipe_ScaleMode_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_mediapipe_ScaleMode_descriptor,
        new java.lang.String[] { });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
